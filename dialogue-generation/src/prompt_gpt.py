'''
    prompt_chatgpt.py -- main entry point, reads console arguments and starts 
    dialog generation
'''

from arguments import Arguments
import logging
from tqdm.auto import tqdm
from dialog_generator import OpenAIDialogGenerator
import os
import sys

# for simplicity, just redirect everything to the console
logging.basicConfig(level=logging.ERROR, 
    handlers=[logging.StreamHandler(sys.stdout)])

if __name__ == "__main__":    
    logger = logging.getLogger(__name__)
    parser = Arguments()
    args = parser.parse_args()

    # check whether output folder exists and create new one if necessary
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    dialgen = OpenAIDialogGenerator(args)

    if not args.fix:
        for i in tqdm(range(args.dialogs)):
            filename = os.path.join(args.output_dir, 
                                    'dialog_'+ str(i) + '.json')
            dialog = dialgen.generate_dialog(filename, logger)
    else:
        for file in tqdm(os.listdir(args.output_dir)):
            filename = os.path.join(args.output_dir, file)
            dialgen.fix_dialog(filename, logger)