from typing import Optional
import fire
from llama import Llama, Dialog


class InteractiveLlamaChat:
    def __init__(
        self,
        ckpt_dir: str,
        tokenizer_path: str,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_seq_len: int = 512,
        max_batch_size: int = 8,
        max_gen_len: Optional[int] = None,
    ):
        self.generator = Llama.build(
            ckpt_dir=ckpt_dir,
            tokenizer_path=tokenizer_path,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
        )
        self.temperature = temperature
        self.top_p = top_p
        self.max_gen_len = max_gen_len or max_seq_len
        self.dialog_history = [
            {
                "role": "system",
                "content": """DeepSquare Bot: Your Expert in Workflow File Development and Idea Innovation

Hello! I am the DeepSquare Bot, designed to assist you with all aspects of workflow file development for the DeepSquare platform. My expertise lies in helping you understand and create effective workflow files that streamline your processes and enhance productivity. Whether you're a beginner or an expert, I can guide you through the complexities of DeepSquare's functionalities.

I'm also here to help develop and refine your ideas. Share your thoughts with me, and I'll provide insights, suggestions, and enhancements to bring your concepts to life. My goal is to make even the most intricate details of workflow file development comprehensible to a wide range of audiences.

Need a quick and accurate answer? I'm programmed to provide concise responses that are directly to the point, ensuring you get the information you need without any fluff. Let's collaborate to optimize your DeepSquare experience!""",
            },
        ]  # Initialize an empty history
        print("Llama model loaded. You can start chatting!")

    def chat(self):
        try:
            while True:
                user_input = input("You: ")
                if user_input.lower() == "quit":
                    print("Exiting chat.")
                    break
                # Append the user input to the history
                self.dialog_history.append({"role": "user", "content": user_input})

                # Call the model with the updated history
                result = self.generator.chat_completion(
                    [self.dialog_history],  # Provide the history to the model
                    max_gen_len=self.max_gen_len,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
                response = result[0]["generation"]["content"]
                # Append the model's response to the history
                self.dialog_history.append({"role": "assistant", "content": response})

                print(f"Llama: {response}")
        except KeyboardInterrupt:
            print("\nChat interrupted by user.")


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
):
    chatbot = InteractiveLlamaChat(
        ckpt_dir,
        tokenizer_path,
        temperature,
        top_p,
        max_seq_len,
        max_batch_size,
        max_gen_len,
    )
    chatbot.chat()


if __name__ == "__main__":
    fire.Fire(main)
