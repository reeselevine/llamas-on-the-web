import { Template } from '@huggingface/jinja';
import { Wllama } from '@reeselevine/wllama-webgpu';
import { DEFAULT_CHAT_TEMPLATE } from './config';

export type ChatMessage = {
  role: 'system' | 'user' | 'assistant';
  content: string;
};

export const formatChat = async (
  modelWllama: Wllama,
  messages: ChatMessage[]
): Promise<string> => {
  const templateStr = modelWllama.getChatTemplate() ?? DEFAULT_CHAT_TEMPLATE;
  const template = new Template(templateStr);
  const bosToken = await modelWllama.detokenize([modelWllama.getBOS()], true);
  const eosToken = await modelWllama.detokenize([modelWllama.getEOS()], true);

  return template.render({
    messages,
    bos_token: bosToken,
    eos_token: eosToken,
    add_generation_prompt: true,
  });
};
