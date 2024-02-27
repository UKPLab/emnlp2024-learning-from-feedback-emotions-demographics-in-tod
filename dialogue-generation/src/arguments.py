import argparse

class Arguments:

    def __init__(self) -> None:
        self.__parser = argparse.ArgumentParser()
        self.__build_args(self.__parser)

    def parse_args(self) -> argparse.Namespace:
        return self.__parser.parse_args()

    def __build_args(self, parser) -> None:
        parser.add_argument("--apipath", 
                            help="Path to your OpenAI api key.",
                            type=str,
                            required=True)
        parser.add_argument("--tasks", 
                            help="Path to your OpenAI api key.",
                            type=str,
                            nargs='+',
                            required=True,
                            default=[])
        parser.add_argument("--topic", 
                            help="Topic to ask questions about (for question answering).",
                            type=str,
                            required=False)        
        parser.add_argument("--debug", 
                            action="store_true",
                            help="If set, will print debug stmts.")
        parser.add_argument("--retries", 
                            type=int, 
                            help="Number of retries in case of error when calling OpenAI.",
                            default=10)
        parser.add_argument("--dialogs",
                            type=int,
                            help="Number of dialogs to generate.",
                            default=1)
        parser.add_argument("--output_dir",
                            type=str,
                            help="Where to dump the generated dialogs.",
                            required=True)  
        parser.add_argument("--config_prompts",
                            type=str,
                            help="Config file with locations for all prompt files",
                            required=False,
                            default="prompt_config.yml")
        parser.add_argument("--errors",
                            type=int,
                            help="The number of errors a generated dialog should include.",
                            required=False)
        parser.add_argument("--fix",
                            action="store_true",
                            help="If set, will fix the first error in each of the dialogs in the output directory.",
                            required=False)        