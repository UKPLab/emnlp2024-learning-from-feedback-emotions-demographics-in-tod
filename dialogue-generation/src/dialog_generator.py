'''
    dialog_generator.py -- Script bundles everything needed for dialog generation in the class OpenAIDialogGenerator. 
'''

import json
import logging
import traceback
import argparse
import re
import os

from typing import List
from model.prompts import PromptResult
from model.dialog_container import DialogContainer
from model.record import Record
from openai_handler import OpenAi
from prompt_generator import PromptGenerator
import utils.errors as errors


class OpenAIDialogGenerator:
    """This class has methods for dialog generation and annotation.
    """

    def __init__(self,
                 args: argparse.Namespace):
        self.args = args
        self.openai = OpenAi(self.args.apipath,
                             self.args.retries)
        self.prompt_gen = PromptGenerator(self.args.config_prompts)

    def _mask(self, dialog: List[Record], subject: str, idx: int) -> str:
        uts = []
        for i, ut in enumerate(dialog):
            pre = f'{"[Person]" if ut.subject != "agent" else "[Robot]"}'
            uts.append(f'{pre}: {ut.text if i != idx else "<mask>"}')
        return '\n\n'.join(uts) if subject.lower() != 'robot'\
            else '\n\n'.join(uts[:idx+1])

    def _fix_dialog(self, container: DialogContainer)\
            -> int | None:            
        dialog = container.dialog.session.records
        documents = container.raw_data.documents
        if len(dialog) > 0:
            try:
                idx_error = [i for i, ut in enumerate(dialog) if
                             ut.error_type != '' and ut.subject == 'agent'][0]
                # get the index of the error with respect to the number of
                # agent utterances in the dialog; important for doing the
                # annotations
                agent_uts = [ut for ut in dialog if ut.subject == 'agent']
                idx_robot_annotations = [i for i, ut in
                                         enumerate(agent_uts) if ut.error_type != ''][0]

                # if the agent started the conversation, the idx of the
                # user utterance where to fix the annotations is the same
                # as that of the robot utterance. If not, it's the idx of
                # the erroneous robot utterance + 1
                if dialog[0].subject == 'agent':
                    idx_user_annotations = idx_robot_annotations
                else:
                    idx_user_annotations = idx_robot_annotations + 1

                task_name = dialog[idx_error].intent

                #
                # generate new robot response
                #

                add_info = ''
                if task_name.lower() == 'question_answering':
                    if idx_error >= 1:
                        if len(dialog[idx_error-1].attachments) > 0:
                            doc_id = dialog[idx_error-1].attachments[0]
                            add_info = documents[doc_id]
                    else:
                        # if the erroneous robot utterance is the first
                        # utterance in the dialog, no document is
                        # available, then use the error scenario as
                        # additional info
                        add_info = dialog[idx_error].error_scenario

                robot_masked = self._mask(dialog, 'Robot', idx_error)
                prompt = self.prompt_gen.fix_dialog_prompt([task_name],
                                                           'robot', robot_masked, [add_info])
                gen_robot_ut =\
                    self.openai.prompt_chatgpt(prompt, 'json')                

                # set new robot response (if available, otherwise stick to
                # old response)
                if 'Utterance' in gen_robot_ut.content:
                    dialog[idx_error].text =\
                        gen_robot_ut.content['Utterance']
                    dialog[idx_error].error_scenario = ''
                    dialog[idx_error].error_type = ''
                    dialog[idx_error].slots = []
                    dialog[idx_error].emotion = ''                    

                    if len(container.dialog.session.records) >= idx_error+1:
                        container.dialog.session.records.pop(idx_error+1)
                    if len(container.dialog.session.records) >= idx_error+1:
                        container.dialog.session.records.pop(idx_error+1)
                    dialog = container.dialog.session.records
                
                return idx_robot_annotations                    
            except IndexError:                
                raise errors.NoErrorsInDialogException(container.filename)

    def _do_annotations(self, dialog: DialogContainer,
                        annotation_idx=None) -> None:

        #idx_user = idx_user[0] if idx_user else None
        idx_robot = annotation_idx if annotation_idx else None

        # generate slots
        for task in self.args.tasks:
            prompt = self.prompt_gen.slot_prompt(task,
                                                 dialog.dialog.stringify())
            gen_slots = self.openai.prompt_chatgpt(prompt, 'json')
            dialog.prompts.slot_generation.append(gen_slots)

        #
        # everything after this point is constant (not task or
        # scenario-related)
        #

        # generate emotions
        prompt = self.prompt_gen.other_annotation_prompt('emotions', dialog)
        gen_emotions = self.openai.prompt_chatgpt(prompt, 'json')

        if not idx_robot:
            dialog.prompts.emotions_generation = gen_emotions

        for _gen_slots in dialog.prompts.slot_generation:
            dialog.session.assign_slot_values(_gen_slots.content,
                                              idx_robot)

        dialog.dump()

    def fix_dialog(self,
                   file: str,
                   logger: logging.Logger) -> None: 
        try:       
            container = DialogContainer.from_json(file)
            annotation_idx = self._fix_dialog(container)            
            if annotation_idx is not None:
                self._do_annotations(container, annotation_idx)        
        except re.error as error:
            print(f'Catched re.error: {error}.')
            print(f'Filename: {file}\n\n\n')
        except errors.NoErrorsInDialogException as error:
            #logger.info('Catched json decoder error. Delete dialog.')
            print(f'NoErrorsInDialogException: {error}.')
            print(f'Stacktrace: {traceback.format_exc()}')            
            print(f'Filename: {file}\n\n\n')
            print('\n\n\n')            
        except Exception as error:
            #logger.info('Catched json decoder error. Delete dialog.')
            print(f'Catched json decoder error: {error}.')
            print(f'Stacktrace: {traceback.format_exc()}')            
            print(f'Filename: {file}\n\n\n')
            print('\n\n\n')

            
    def generate_dialog(self,
                        outfile: str,
                        logger: logging.Logger) -> None:

        dialog = DialogContainer(outfile, None)

        # init to solve warning in catch block (unbound variables)
        prompt = ''
        gen_dialog = ''

        try:
            scenarios = PromptResult()
            if self.args.errors:
                scenario_prompt, _documents =\
                    self.prompt_gen.error_scenario_prompt(self.args.errors,
                                                            self.args.tasks, self.args.topic)
                scenarios = self.openai.prompt_chatgpt(scenario_prompt,
                                                        'json')
                error_dialog_prompt, persona, docs =\
                    self.prompt_gen.error_dialog_prompt(scenarios.content,
                                                        self.args.tasks, _documents)
                gen_dialog =\
                    self.openai.prompt_chatgpt(error_dialog_prompt,
                                                'dialog')
            else:
                prompt, persona, docs =\
                    self.prompt_gen.dialog_prompt(self.args.tasks,
                                                    self.args.topic)
                gen_dialog = self.openai.prompt_chatgpt(prompt, 'dialog')

            dialog = DialogContainer(filename=outfile,
                                        response=gen_dialog.open_ai_response,
                                        scenarios=scenarios.content)
            dialog.prompts.dialog_generation = gen_dialog
            dialog.raw_data.persona = persona
            dialog.raw_data.documents = docs
            if self.args.errors:
                dialog.raw_data.prompts.error_scenarios = scenarios
            if dialog is None:
                raise errors.CallOpenAIException()

            self._do_annotations(dialog)
            # import pdb; pdb.set_trace()

        except re.error as error:
            print(f'Catched re.error: {error}. Delete dialog.')
            dialog.delete()
        except Exception as error:
            #logger.info('Catched json decoder error. Delete dialog.')
            print(f'Catched json decoder error: {error}.')
            print(f'Stacktrace: {traceback.format_exc()}')
            print('Delete dialog.')
            print('\n\n\n')
            dialog.delete()
        # except (TypeError, AttributeError, KeyError) as error:
            #logger.info('Catched json decoder error. Delete dialog.')
            #print(f'Catched json decoder error: {error}.')
            #print(f'Stacktrace: {traceback.format_exc()}')
            #print('Delete dialog.')
            # dialog.delete()
        # except errors.CallOpenAIException as error:
            #logger.info('Catched json decoder error. Delete dialog.')
            #print(f'Catched json decoder error: {error}.')
            #print(f'Stacktrace: {traceback.format_exc()}')
            #print('Delete dialog.')
            # dialog.delete()
