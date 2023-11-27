from typing import Optional
import fire
from llama import Llama, Dialog

class InteractiveLlamaChat:
    def __init__(self, ckpt_dir: str, tokenizer_path: str, temperature: float = 0.6, top_p: float = 0.9, max_seq_len: int = 512, max_batch_size: int = 8, max_gen_len: Optional[int] = None):
        self.generator = Llama.build(
            ckpt_dir=ckpt_dir,
            tokenizer_path=tokenizer_path,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
        )
        self.temperature = temperature
        self.top_p = top_p
        self.max_gen_len = max_gen_len or max_seq_len
        print("Llama model loaded. You can start chatting!")

    def chat(self):
        try:
            while True:
                user_input = input("You: ")
                if user_input.lower() == 'quit':
                    print("Exiting chat.")
                    break
                dialog = [{"role": "user", "content": user_input}]
                result = self.generator.chat_completion(
                    [dialog],  # type: ignore
                    max_gen_len=self.max_gen_len,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
                response = result[0]['generation']['content']
                print(f"Llama: {response}")
        except KeyboardInterrupt:
            print("\nChat interrupted by user.")

def main(ckpt_dir: str, tokenizer_path: str, temperature: float = 0.6, top_p: float = 0.9, max_seq_len: int = 512, max_batch_size: int = 8, max_gen_len: Optional[int] = None):
    chatbot = InteractiveLlamaChat(ckpt_dir, tokenizer_path, temperature, top_p, max_seq_len, max_batch_size, max_gen_len)
    chatbot.chat()

if __name__ == "__main__":
    fire.Fire(main)
