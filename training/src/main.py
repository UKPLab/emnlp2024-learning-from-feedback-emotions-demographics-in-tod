import torch
import os
import time
import re

from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    get_scheduler,
    LlamaForCausalLM,
    BitsAndBytesConfig
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from datasets.poste import PosteDataset
from datasets.poste_instructions import PosteDatasetInstructions
from logger import DataLogger
from metrics import n_gram_overlapping_f1
from arguments import Arguments
from pathlib import Path
from typing import Tuple, Generator


class Trainer():

    def __init__(self, args):

        self.args = args

        # create accelerator to take care of whether to use one or
        # more GPUs or one CPU
        if not "LLAMA" in self.args.pretrained_model.upper():
            self.accelerator = Accelerator(mixed_precision=self.args.mixed_prec)
        else:
            self.accelerator = Accelerator()
        self.device = self.accelerator.device
        print(f"Device: {self.device}")

        # load tokenizer and add special tokens, load model
        additional_tokens = [
            "<user_reaction>",
            "<error_text>",
            "<intent>",
            "<slots>",
            "<knowledge>",
            "<dialog>",
            "<system>",
            "<user>",
            "<user_persona>",
            "<gender>",
            "<name>",
            "<language>",
            "<age>",
            "<job>",
            "<user_emotion>",
            "<user_gesture>",
            "<system_action>",
            "<outcome_operation>",
            "<bill_form_payment_procedure>",
            "<import_payment>",
            "<destination>",
            "<type_of_bills>",
            "<host_name>",
            "<confirmation_to_open_the_turnstile>",
            "<delivery_option>",
            "<ticket_number>",
            "<verification_call>",
            "<weight>",
            "<phone_number>",
            "<meeting_date_and_time>",
            "<bill_form_name>",
            "<shipping_box_description>",
            "<host_email>",
            "<shipping_procedure>",
            "<meeting_room_identifier>",
            "<guest_name>",
            "<confirmation_to_open_turnstile>",
            "<phone_provider>",
            "<package_required>",
            "<alternative_host_email>",
            "<bill_form_description>",
            "<question>",
            "<type_of_service>",
            "<alternative_host_name>",
            "<shipping_box_name>",
            "<shipping_time>",
            "<evidence>",
        ]

        if "T5" in self.args.pretrained_model.upper():
            self.tokenizer =\
                AutoTokenizer.from_pretrained(self.args.pretrained_model)
            add_special_toks = self.tokenizer.special_tokens_map[
                "additional_special_tokens"
            ]
            add_special_toks += additional_tokens
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": add_special_toks}
            )
            model = T5ForConditionalGeneration.from_pretrained(
                self.args.pretrained_model
            )
        elif "GPT2" in self.args.pretrained_model.upper():
            self.tokenizer =\
                GPT2Tokenizer.from_pretrained(self.args.pretrained_model,
                    additional_special_tokens=additional_tokens,
                    add_prefix_space=True)
            self.tokenizer.pad_token = self.tokenizer.eos_token    
            self.tokenizer._add_tokens(additional_tokens, special_tokens=True)
            model = GPT2LMHeadModel.from_pretrained(self.args.pretrained_model,
                pad_token_id=self.tokenizer.eos_token_id)
        elif "LLAMA" in self.args.pretrained_model.upper():            
            self.tokenizer =\
                AutoTokenizer.from_pretrained(self.args.pretrained_model)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=False,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )

            model =\
                LlamaForCausalLM.from_pretrained(self.args.pretrained_model, 
                    quantization_config=bnb_config)
            model.resize_token_embeddings(len(self.tokenizer))
            model.gradient_checkpointing_enable()
            model = prepare_model_for_kbit_training(model)
            config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                    "lm_head",
                ],
                bias="none",
                lora_dropout=0.05,  # Conventional
                task_type="CAUSAL_LM",
            )

            model = get_peft_model(model, config)
            self.print_trainable_parameters(model)
        else:
            raise Exception("Unknown model class!")

        # adjust embeddings
        model.resize_token_embeddings(len(self.tokenizer))
        model.config.use_cache = False

        optimizer = None
        lr_scheduler = None
        train_dataloader = None
        eval_dataloader = None

        # build dataloaders
        if self.args.train_data:
            print("Script initialized with data - will do training.")
            train_dataloader =\
                self.create_dataloader(Path(self.args.train_data), 'train')
            optimizer, lr_scheduler = self.get_optimizer_scheduler(
                model.parameters(), len(train_dataloader), self.args.epochs
            )
        if self.args.eval_data:
            print("Script initialized with evaluation data" + " - will do evaluation.")
            eval_dataloader =\
                self.create_dataloader(Path(self.args.eval_data), 'eval')
        if self.args.test_data:
            print("Script initialized with test data - will do testing")
            test_dataloader =\
                self.create_dataloader(Path(self.args.test_data), 'eval')

        # activate gradient checkpointing
        if self.args.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        # let accelerator take care of the distribution
        (
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.eval_dataloader,
            self.test_dataloader,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            model,
            optimizer,
            train_dataloader,
            eval_dataloader,
            test_dataloader,
            lr_scheduler,
        )

        # initialize tensorboard logging
        self.writer = SummaryWriter()

        # initialize file logging
        self.data_logger = DataLogger(self.args.data_log, 
            self.args.pretrained_model.upper(), self.tokenizer)

    def print_trainable_parameters(self, model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )            

    def get_optimizer_scheduler(
        self, model_parameters: Generator, len_dataloader: int, epochs: int
    ) -> Tuple[AdamW, LambdaLR]:

        optimizer = AdamW(model_parameters, lr=5e-5)
        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=epochs * len_dataloader,
        )
        return optimizer, lr_scheduler

    def create_dataloader(self, data_path: Path, _type:str):

        _pin_memory = True
        if "LLAMA" in self.args.pretrained_model.upper():            
            dataset = PosteDatasetInstructions(data_path, self.tokenizer,
                _type, self.args)
            _pin_memory = False
        else:
            dataset = PosteDataset(data_path, self.tokenizer, 
                self.args.pretrained_model.upper(), _type, self.args)
        

        dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.args.num_workers,
            pin_memory=_pin_memory,
        )

        print(
            f"Number of samples: {len(dataset)}; Number of batches"
            + f" {len(dataloader)}"
        )
        return dataloader

    def save(self, epoch: int, f1: float):
        model_path = self.args.model_path
        if not os.path.exists(self.args.model_path):
            os.makedirs(model_path)

        # wait for every other process to finish before saving
        self.accelerator.wait_for_everyone()

        model_name = f'{self.args.experiment_name}_{epoch}_f1_{f1}'
        model = self.accelerator.unwrap_model(self.model)
        model.save_pretrained(os.path.join(model_path, model_name), 
            from_pt=True)
        
        if "LLAMA" in self.args.pretrained_model.upper():            
            self.tokenizer.save_pretrained(os.path.join(model_path, model_name, 
                "lora")) 
            
            # reload the model and merge with lora weights
            # (for saving the complete model as finetuned model)
            time.sleep(5)
            base_model = LlamaForCausalLM.from_pretrained(
                self.args.pretrained_model,
                low_cpu_mem_usage=True,
                return_dict=True,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
            )
            base_model.resize_token_embeddings(len(self.tokenizer))
            model = PeftModel.from_pretrained(
                base_model, os.path.join(model_path, model_name, "lora")
            )
            model = model.merge_and_unload()

            self.tokenizer.save_pretrained(
                os.path.join(model_path, model_name, "finetuned")
            )
            model.save_pretrained(os.path.join(model_path, model_name, "finetuned"))
        else:
            self.tokenizer.save_pretrained(os.path.join(model_path, model_name))


    def evaluate(self, dataloader: DataLoader, mode: str, epoch: int = 0):
        print(f"\nStarting {mode}.")
        metrics = {"prec": 0.0, "recall": 0.0, "f1": 0.0}
        progress_bar = tqdm(range(len(dataloader)))

        self.model.eval()
        for i, batch in enumerate(dataloader):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                if "T5" in self.args.pretrained_model.upper():
                    generations =\
                        self.accelerator.unwrap_model(self.model).\
                            generate(input_ids=batch["input"].squeeze(),
                                     max_length=self.args.max_target_token)
                else:
                    generations =\
                        self.accelerator.unwrap_model(self.model)\
                            .generate(input_ids=batch['input'].squeeze(),
                                max_new_tokens=300)
                    
            prec, recall, f1 = self.calc_metrics(batch, generations)
            metrics["prec"] += prec
            metrics["recall"] += recall
            metrics["f1"] += f1
        
            # log to progress bar
            progress_bar.set_postfix(
                {"prec": prec, "recall": recall, "f1": f1}
            )
            progress_bar.update(1)

            # log to tensorboard
            iteration = i if mode == "test" else i + len(dataloader) * epoch
            self.writer.add_scalar("Precision", prec, iteration)
            self.writer.add_scalar("Recall", recall, iteration)
            self.writer.add_scalar("F1", f1, iteration)            

            # log to file
            if i % 100 == 0:
                self.data_logger.log_eval_test(batch, generations, mode)

        metrics["prec"] = round(metrics["prec"] / len(dataloader), 3)
        metrics["recall"] = round(metrics["recall"] / len(dataloader), 3)
        metrics["f1"] = round(metrics["f1"] / len(dataloader), 3)
        
        # save model
        self.save(epoch, metrics["f1"])

        # log to console
        print(f"Finished {mode}.")
        print(
            f'Result: Precision: {metrics["prec"]}, Recall: '
            + f'{metrics["recall"]}, F1: {metrics["f1"]}'
        )

    def calc_metrics(self, batch: torch.Tensor, generations: torch.Tensor):

        predictions = self.tokenizer.batch_decode(generations, 
            skip_special_tokens=False)
        gold =\
            self.tokenizer.batch_decode(batch["target"].squeeze(), 
                skip_special_tokens=False)
        
        if "T5" in self.args.pretrained_model.upper():
            predictions = [pred.strip() for pred in predictions]
            golds = [pred.strip() for pred in gold]
        elif "GPT2" in self.args.pretrained_model.upper():
            predictions = [pred.split('<intent>')[0] if '<intent>' in pred 
                else pred for pred in predictions]
            predictions = [pred.replace('<|endoftext|>', '').strip() 
                for pred in predictions]
            golds = [pred.replace('<|endoftext|>', '').strip() 
                for pred in gold]    
        elif "LLAMA" in self.args.pretrained_model.upper():
            predictions = [re.sub(r"<\|begin_of_text\|>|<\|end_of_text\|>", "",
                pred) for pred in predictions]
            predictions = [pred.split("<intent>")[0] if "<intent>" in pred 
                else pred for pred in predictions]
            predictions = [pred.replace("<|endoftext|>", "").strip() 
                for pred in predictions]
            gold = [re.sub(r"<\|begin_of_text\|>|<\|end_of_text\|>", "", g) 
                for g in gold]
            golds = [pred.replace("<|endoftext|>", "").strip() for pred in gold]

        prec, recall, f1 = n_gram_overlapping_f1(predictions, golds,
            self.tokenizer)        

        return prec, recall, f1

    def main(self):
        if self.args.train_data:
            for epoch in range(self.args.epochs):
                print(f"\nStarting epoch {epoch}")

                self.model.train()
                self.freeze_transformer(epoch)
                progress_bar = tqdm(range(len(self.train_dataloader)))

                for i, batch in enumerate(self.train_dataloader):
                    batch = {k: v.to(self.device) for k, v in batch.items()}

                    # all labels set to -100 are ignored; the loss is only
                    # calculated for the other tokens.
                    labels = batch["target"].clone()
                    labels[batch["target"][:, :] == self.tokenizer.pad_token_id] = -100

                    outputs = self.model(
                        input_ids=batch["input"].squeeze(),
                        labels=labels.squeeze(),
                        return_dict=True,
                    )
                    loss = outputs["loss"]
                    self.accelerator.backward(loss)

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                    # logging progress bar
                    progress_bar.set_postfix({"loss": round(loss.item(), 3)})
                    progress_bar.update(1)

                    # logging tensorboard
                    self.writer.add_scalar(
                        "Loss", loss, i + len(self.train_dataloader) * epoch
                    )

                    # logging to file
                    if i % 100 == 0:
                        self.data_logger.log_train(batch)

                if self.args.eval_data:
                    self.evaluate(self.eval_dataloader, "evaluation", epoch)

        if self.args.test_data:
            self.evaluate(self.test_dataloader, "test")


if __name__ == "__main__":

    parser = Arguments()
    trainer = Trainer(parser.parse_args())
    trainer.main()
