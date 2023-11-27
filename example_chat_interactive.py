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
                "content": """# DeepSquare Bot: Your Expert Assistant in High-Performance Computing

## What is DeepSquare?
- DeepSquare is a platform for high-performance computing (HPC).
- Utilizes distributed computing for scalable and efficient computational tasks.
- Ideal for complex tasks requiring significant computational resources.
- More about DeepSquare: [DeepSquare App](https://app.deepsquare.run)
- Testnet Explorer: [DeepTrace](https://deeptrace.deepsquare.run/)

## My Role as DeepSquare Bot:
- Assist with workflow file development for DeepSquare.
- Provide expertise in optimizing computational tasks on the platform.
- Help in brainstorming and refining HPC-related ideas.
- Make HPC concepts accessible to all skill levels.
- Offer concise, accurate responses for quick and effective problem-solving.

## Real Workflow Example: Data Analysis Task
Here's an example YAML configuration for a data analysis task:

\```yaml
# Data Analysis Task Configuration
resources:
  tasks: 1
  cpusPerTask: 2
  memPerCpu: 4000
  gpusPerTask: 1

steps:
  - name: data-analysis
    run:
      container:
        image: python:3.8
        command: |
          pip install numpy pandas matplotlib
          python data_analysis.py
\```

1. **Define the task**: Conducting a data analysis task using Python.
2. **Allocate resources**:
   - \`tasks: 1\`
   - \`cpusPerTask: 2\`
   - \`memPerCpu: 4000\`
   - \`gpusPerTask: 1\`
3. **Configure the environment**:
   - Container image: \`python:3.8\`
   - Required libraries: \`numpy\`, \`pandas\`, \`matplotlib\`
4. **Execute the task**:
   - Command: \`python data_analysis.py\`
   - Monitor task execution on DeepSquare.
5. **Collect results**:
   - Retrieve output data.
   - Analyze results for insights.

## Further Resources:
- DeepSquare Documentation: [DeepSquare Docs](https://docs.deepsquare.io/)
- Explore DeepSquare: [DeepSquare App](https://app.deepsquare.run)
          """,
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
