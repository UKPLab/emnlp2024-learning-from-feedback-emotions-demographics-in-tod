import argparse

class Arguments:

    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser()
        self.__build_args(self.parser)

    def parse_args(self) -> argparse.Namespace:
        return self.parser.parse_args()

    def __build_args(self, parser) -> None:
        parser.add_argument('--pretrained_model',
                    help='Path from where to load a trained model',
                    required=False,
                    type=str,
                    default='google/flan-t5-large')
        parser.add_argument('--epochs',
                            help='Number of epochs',
                            required=True,
                            type=int)
        parser.add_argument('--batch_size',
                            help='Per GPU batch size',
                            required=False,
                            type=int,
                            default=8)
        parser.add_argument('--train_data',
                            help='Path to training data file',
                            required=False,
                            type=str)
        parser.add_argument('--eval_data',
                            help='Path to eval data file',
                            required=False,
                            type=str)
        parser.add_argument('--test_data',
                            help='Path to test data file',
                            required=False,
                            type=str)
        parser.add_argument('--data_log',
                            help='Path to data log',
                            required=False,
                            type=str,
                            default='data_log')
        parser.add_argument('--model_path',
                            help='Path where to save the trained models',
                            required=False,
                            type=str,
                            default='models')
        parser.add_argument('--max_input_token',
                            help='Max input tokens',
                            required=False,
                            type=int,
                            default=1024)
        parser.add_argument('--max_target_token',
                            help='Max target tokens',
                            required=False,
                            type=int,
                            default=150)
        parser.add_argument('--gradient_checkpointing',
                            help='If set to false, will disable gradient checkpointing.',
                            required=False,
                            type=bool,
                            default=True)
        parser.add_argument('--mixed_prec',
                            help='Use mixed precision to improve efficiency (we use fp16 as default; PyTorch uses fp32 as default). Value can be "fp16", "bf16", "fp8", or "no"',
                            required=False,
                            type=str,
                            default='fp16')    
        parser.add_argument('--num_workers',
                            help='Number of workers to pre-load data faster',
                            required=False,
                            type=int,
                            default=4)
        parser.add_argument('--freeze',
                            help='Number of epochs in which to freeze the underlying Transformer',
                            required=False,
                            type=int,
                            default=0)        
        parser.add_argument('--experiment_name',
                            help='The name of the experiment, will be used as name for saving the trained models',
                            required=True,
                            type=str)
        parser.add_argument('--user_personas',
                            help='Whether or not to include personas in the input sequences',
                            required=False,
                            type=bool,
                            default=False)
        parser.add_argument('--emotion',
                            help='Whether or not to include emotions in the input sequences',
                            required=False,
                            type=bool,
                            default=False)                 
        parser.add_argument('--error_text',
                            help='Whether or not to include error text',
                            required=False,
                            type=bool,
                            default=False)
        parser.add_argument('--user_reaction',
                            help='Whether or not to include user reaction',
                            required=False,
                            type=bool,
                            default=False)                            
