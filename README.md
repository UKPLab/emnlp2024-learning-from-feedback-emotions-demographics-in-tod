# Learning from Implicit User Feedback, Emotions and Demographic Information in Task-Oriented and Document-Grounded Dialogues
[![Arxiv](https://img.shields.io/badge/Arxiv-2401.09248-red?style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2401.09248)
[![License](https://img.shields.io/github/license/UKPLab/ukp-project-template)](https://opensource.org/licenses/Apache-2.0)
[![Python Versions](https://img.shields.io/badge/Python-3.10-blue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)

In this repository, we provide the generation framework and example scripts for using the FEDI data. FEDI is the first task-oriented document-grounded dialogue dataset for learning from demographic information, user emotions and implicit user feedback. In its current version, FEDI consists of 8,852 dialogues, divided into 1,988 feedback-free dialogues, including 326 test dialogues, and 6,864 feedback dialogues (1,716 in four versions, each with one feedback scneario less per dialogue). The original dataset is available [here](https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/4181).

<p align="center">
<img align="center" src="resources/dialogue_example.png" alt="drawing" width="300"/>
</p>

## FEDI v2
We are delighted to provide a new and updated version of FEDI, [FEDI v2](https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/4554). It fixes some of the issues resulting from synthetic data generation and extends the tasks and domains covered in the original dataset. In detail, FEDI v2 improves FEDI v1 as follows:

1. FEDI v2 addresses the following issues in FEDI v1:
    - Annotations for user emotions frequently use the _neutral_ signal and do not fit the dialogue context.
    - System utterances sometimes include hallucinated user names.
    - Task-oriented dialogues sometimes include placeholders for slot values, especially in the case of _parcel choice_ and in situations where not all slots are required to sucessfully complete a task, e.g., if the user already has a shipping box and just requires information about the shipping procedure.
    - Knowledge-grounded dialogues frequently include hallucinated annotations from foreign domains.

2. We introduce _parcel shipping_ as a new task, with the goal to guide the user thorugh the process of parcel shipping from choosing the correct shipping box to selecting a proper shipping product. to fulfill this task, the agent requires the following information: destination, weight, dimensions (of the parcel), delivery option. Based on the provided information, the agent gives advice regarding the best shipping product to choose.

3. We extent the domains for knowledge-grounded question answering dialogues with the following domains:
    - _Energy_: This domain covers questions related to _Poste Energia_ which is an Italian electricity and gas provider.
    - _Mail_: this domain includes questions related to senting and receiving registered or insured mail, letters, and telegrams. In addition, it covers the Ialian _Follow Me_ service and personal post office boxes.
    - _Parcel_: This domain encompasses questions converning the sending and receiving procedures of parcels with the Italian post.
    - _Prepaid Services_: This domain consists of questions related to different prepaid cards available in Italy provided by _PostePay_, an Italian company offering a prepaid card service.

For each new domain and the task of parcel shipping, FEDI v2 provides 240 feedback-free and 960 feedback-annotated (240 dialogues in four versions). The new dialogues were generated using GPT-4o, the old dialogues were improve following a semi-automatic approach.

FEDI v2 was created by Nils Bez as part of his bachelor's thesis, [Learning from Feedback Situations in Task-Oriented Dialogues](https://www.informatik.tu-darmstadt.de/ukp/teaching_ukp/ukp_teaching_theses/ukp_theses_completed/index.en.jsp), supervised by Dominic Petrak and Iryna Gurevych. 

## Dataset Description
FEDI covers four use cases for task-oriented document-grounded dialogue systems from three domains:

- __Post Office Services:__ For post office services, FEDI includes (1) customer support for parcel shipping, i.e., guiding them through the process of parcel shipping from choosing the right shipping box to informing them about the approximate delivery time, and (2) topping up a prepaid SIM card. Following are the slot values for this task:
  - Parcel Shipping:
    - Destination (informable) -- The city and country of destination; national or international.
    - Weight (informable) -- The weight of the item to be shipped, lightweight (up to 5kg), average (up to 20kg), heavy (up to 30kg).
    - Package required (informable) -- Whether or not a new shipping box is required.
    - Delivery option (informable) -- Express or standard delivery.
    - Country of destination (informable) -- The destination country.
    - Shipping box name (requestable) -- Name of the best suitable shipping box (small-sized, medium-sized, large-sized), based on the weight of the item to be sent.
    - Shipping box description (requestable) -- Brief description on why the suggested shipping box is the best option.
    - Shipping procedure (requestable) -- Description of the shipping procedure (e.g., take the box to the counter...).
    - Shipping time (requestable) -- Expected delivery time, one to three days for national, four to six days for european, and 3-4 weeks for international deliveries.
  - Top Up SIM Card:
    - Phone number (informable) -- Table or mobile phone number with country code.
    - Phone provider (informable) -- The prone provider, e.g., Vodafone, POSTE Mobile, ...
    - Import payment (informable) The recharge amount, e.g., 10 euro, 20 euro, 30 euro.
    - Outcome operation (requestable) -- If all required information were provided, the system asks the user to insert the card for payment.
  -  Request Ticket:
      - Type of service (informable) -- The type of service for which the user wants to request support, i.e., parcel shipping or topping up a prepaid SIM card.
      - Ticket number (requestable) -- The ticket number generated for the request.
- __Receptionist Service:__ As receptionist service, FEDI includes access control, i.e., the reception and registration of new visitors in office buildings. Following are the slot values:
  - Guest name (informable) -- The name of the person who wants to access the building.
  - Host name (informable) -- The name of the person the guest wants to visit.
  - Host e-mail (informable) -- The e-mail address of the host.
  - Alternative host name (informable) -- An alternative host, e.g., in case the host is not available.
  - Alternative host e-mail (informable) -- E-mail address of the alternative host.
  - Meeting date and time (informable) -- Date and time of the appointment.
  - Meeting room identifier (informable) -- Unique identifier of the room where the meeting will take place.
  - Verification call (requestable) -- The system can set up a verification call to let the host visually inspect the guest and authorize access.
  - Confirmation to open turnstile (requestable) -- This is a signal to the system that controls the turnstile to let the guest enter.
  - Add. safety information (requestable) -- Any additional safety information, e.g., related to Covid-19.
- __Customer Service in the Insurance Domain:__ As customer service in the insurance domain, FEDI includes question answering in the context of financial topics and pet, health and heritage insurance. These dialogues additionally provide document annotations. Following are the slot values:
  - Question (informable) -- A question to one of the topics.
  - Type of bills (informable) -- If the user asks a question regarding a specific payment slip, they need to provide the type.
  - Evidence (requestable) -- The answer to the user's question.
  - Bill form description (requestable) -- Description of the specific payment form (if the question was about a payment form).
  - Bill form name (requestable) -- Name of the payment form (if the question was about a payment form).
  - Bill form paymend procedure (requestable) -- Information on how to fill the payment form (if the question was about a payment form).

We provide more details on the task descriptions in Appendix A of our paper.
### Problem formulation
We define a dialogue as a set of multiple turns $T$. Each turn consists of two utterances, a user utterance $U_t$ and a system utterance $S_t$. Given the dialogue context $C=[T_0, ..., T_{t-1}]$, and additional information $K$, the task is to predict the user intent $I_t$, generate belief state $B_t$ and system utterance $S_t$:

$(I_t, B_t, S_t) = \text{generate}(K, C, U_t)$

Depending on whether knowledge from a document $D_t$ is required to generate $S_t$ or the user emotion $E_t$, demographic information $DI$, generation error $GE_t$, or implicit user feedback  $F_t$ should be considered, $K=\{D_t, DI, E_t, GE_t, F_t\}$. $DI$ includes the user's gender, age range, occupation, name, and language style. Belief state $B_t$ includes the slot values inferred from the dialogue context $C$.

### Dataset Structure
We provide FEDI in the _dataset_ folder of this repository. It contains the dialogues in a format ready for training and inference. Each dialogue file contains a list of samples generated from the original dialogue. As introduced in the paper, we further distinguish between _feedback-free_ and _feedback_ dialogues. The _feedback_ dialogues are organized in _stages_ (_version_ in the paper). Following is the structure of the data:

```json
{
    "filename": "<path and file identifier>",
    "dialogue": {
        "utterances": [
            {
                "intent": "",
                "slots": [
                    {
                        "span_type": "<slot name>",
                        "span": "<slot value>",
                        "span_character_start_position": "<starting position of the slot value in the text sequence>",
                    },
                ],
                "text": "<utterance>",                
                "emotion": "<emotion annotation>",
                "error_type": "<error type (only available in feedback dialogue system utterances)>",
                "error_scenario": 
                    {
                        "scenario": "<textual description of the error scenario (only available in feedback dialogue system utterances)>",
                        "error_type": "<source error type (only available in feedback dialogue system utterances)>",
                        "user_reaction_type: "<source user reaction type (only available in feedback dialogue system utterances)>"
                    },
                "user_reaction_type": "<assigned user reaction type (only available in feedback dialogue user utterances)>",
                "documents": [
                    ">plain texts from document sources (if any)>"
                ],
                "role": "<user or system, the speaker>"
            },            
        ],
        "background_story": "<the background story of this dialogue (why does it happen?)>",
        "demographics":
            {
                "gender": "<male / female>",
                "job": "<job title>",
                "age": "<age range>",
                "name": "<the name of the user>",
                "language": "<the language style of the user>"
            }
    }
}
```
The feedback dialogues additionally contain the feedback annotations. Another difference is that they do not provide a list of samples per file, but only one sample per file. The _test_ data in the _feedback-free_ folder are also the test data for the feedback dialogues.

## Dialogue and Annotation Generation Framework
We provide our code for dialogue generation and annotation in the _dialogue-generation_ folder. We provide a detailed description of the steps involved in the paper. To run it, please install the packages from our _requirements.txt_ and the _dialogue-generation_ folder itself first using the following commands:

```bash
pip install -r requirements.txt
pip install -e dialogue-generation
```

We used Python 3.10 for dialogue generation. The following figure gives an overview of the folder structure and dependencies:

<p align="center">
<img align="center" src="resources/dialog_generation.png" alt="drawing" width="800"/>
</p>

_prompt_gpt.py_ is the starting script. It initializes dialogue generation according to the arguments passed in the command line and creates a _dialog_generator_ object. This object in turn initializes a session with the OpenAI API and manages the dialog generation process. The dialogs, their annotations, generated prompts and the responses from OpenAI are represented by objects from the classes in the _model_ folder. _prompts_ includes all the hard-coded instructions and additional data, such as names, occupations, and the documents (in txt files). Prompts are generated by the _prompt_generator_ object. _utils_ contains error classes and utility functions (such as for data cleansing).

To run the framework, you have to call _prompt_gpt.py_ using the the following command:

```bash
python prompt_chatgpt.py \
    --apipath <path to your openai api key> \
    --retries <number of retries> \
    --dialogs <number of dialogs> \
    --tasks  <task> \
    --output_dir <directory for finished dialogs> \
    --config_prompts prompt_config.yml
```

For question answering, you have to add the _topic_ argument to the list (e.g., `--topic finance`, with finance as document directory from _prompt_config.yml_). For feedback dialogue generation, you have to add the _errors_ argument, e.g., `--errors 3` to generate a dialogue with three feedback situations. We describe all arguments in _argument.py_.

## Code for Training the Models
This is a condensed version of the code we used for training the models for the paper. We tested it with all models in a Python 3.10. environment and did not observe any issues. If it does not work for you, please reach out!

For running training, just call ```main.py``` like so:

```shell
python main.py  \
    --pretrained_model [path from where to load a trained model] \
    --model_path [path where to save the trained models] \
    --epochs [number of epochs for training] \
    --train_data [path to training data file] \
    --eval_data [path to eval data file] \
    --test_data [path to test data file] \
    --data_log [where to log the data (needs to be created in advance)] \
    --batch_size [per GPU batch size] \
    --max_input_token [max input tokens] \
    --max_target_token [max target tokens] \    
    --mixed_prec [Use mixed precision to improve efficiency (we use fp16 as default; PyTorch uses fp32 as default). Value can be "fp16", "bf16", "fp8", or "no"] \
    --num_workers [Number of workers to pre-load data faster] \
    --experiment_name [The name of the experiment, will be used as name for saving the trained models]
```

For default values, please refer to ```arguments.py``` Besides the parameters mentioned above, it offers the following options:

| Parameter | Description |
|-----------|-------------|
| gradient_checkpointing | If set to false, will disable gradient checkpointing (default true for Llama) |
| freeze | Number of epochs in which to freeze the underlying Transformer |
| user_personas| Whether or not to include personas in the input sequences|
| emotion | Whether or not to include emotions in the input sequences |
| error_text | Whether or not to include error text |
|user_reaction | Whether or not to include user reaction |

Please refer to the paper for the input data format.

## Citation

Please reference our work as follows:

```
@inproceedings{petrak-etal-2024-learning,
    title = "Learning from Implicit User Feedback, Emotions and Demographic Information in Task-Oriented and Document-Grounded Dialogues",
    author = "Petrak, Dominic  and
      Tran, Thy  and
      Gurevych, Iryna",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-emnlp.264",
    pages = "4573--4603",
    abstract = "Implicit user feedback, user emotions and demographic information have shown to be promising sources for improving the accuracy and user engagement of responses generated by dialogue systems. However, the influence of such information on task completion and factual consistency, which are important criteria for task-oriented and document-grounded dialogues, is not yet known. To address this, we introduce FEDI, the first English task-oriented and document-grounded dialogue dataset annotated with this information. Our experiments with Flan-T5, GPT-2 and Llama 2 show a particularly positive impact on task completion and factual consistency. Participants in our human evaluation reported that the responses generated by the feedback-trained models were more informative (Flan-T5 and GPT-2), relevant and factual consistent (Llama 2).",
}
```

## Contact Persons

Dominic Petrak (<petrak@ukp.informatik.tu-darmstadt.de>)
  
## Links

[UKP Lab Homepage](https://www.ukp.tu-darmstadt.de/) | [TU Darmstadt Website](https://www.tu-darmstadt.de/index.en.jsp)

## Disclaimer

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication. 
