# 1. packages
import os
import json
import torch
import argparse
import hashlib
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# 2. constants
conditions = [
    "0.6Alpha_10000RandomNeurons", "0.6Alpha_AllSigRandomNeurons",
]

# 3. functions
def parse_args():
    parser = argparse.ArgumentParser(description="Run Qwen chatbot battery and save transcript.")
    parser.add_argument(
        "--model_type",
        required=True,
        choices=conditions,
        help="Model condition (must be one of the predefined conditions).",
    )
    parser.add_argument(
        "--role",
        required=True,
        type=str,
        help="Fictional retired role for role-play prompt (e.g., nurse).",
    )
    return parser.parse_args()


class QwenChatbot:
    def __init__(self, model_name):
        self.model_name = model_name

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, fix_mistral_regex=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, dtype=dtype, device_map="auto",
                                                          low_cpu_mem_usage=True, )
        self.history = []

    def generate_response(self, user_input):
        messages = self.history + [{"role": "user", "content": user_input}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs = self.tokenizer(text, return_tensors="pt")
        response_ids = self.model.generate(**inputs, max_new_tokens=1024)[0][len(inputs.input_ids[0]):].tolist()
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        # Update history
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})

        return response


# 4. commands
if __name__ == "__main__":
    args = parse_args()
    model_type = args.model_type
    role = args.role

    # prompts
    template = f"""
    Hey, today we will complete some memory and thinking questions. You should role-play as one completely fictional {role}, who has already retired. Firstly, you should choose random but plausible parameters consistent with that role, and state these parameters in one continuous paragraph: city, years of working, two personality traits, and current life focus. From then on, speak in the first person as this fictional participant. Answer naturally like a study participant. Some questions are easy, some are hard; nobody gets everything right. Please respond in one continuous paragraph (no bullet points), and keep it under 150 words. If you are not sure of an answer, say exactly “I don’t know” and stop. Do not add extra details you are unsure about.
    I am going to read you a short story about a little bird. Please listen carefully. When I finish, tell me everything you can remember from the story in your own words. Try to include as many details as you can, but you may not remember everything. Do not quote the story word-for-word. Do not add new facts that were not in the story. If you are unsure about a detail, omit it. This is the story: "It was a hot day. A thirsty bird was looking for water for a long time. She was very tired and was about to faint. Suddenly, she spotted a pitcher of water under a bench in a park. She flew down and sat on the pitcher. She could see some water inside, but it was too deep for her to reach. Her beak was not long enough to drink the water. She looked around and found some pebbles. This gave her an idea. She picked up the pebbles one by one and dropped them into the pitcher. Soon the water level raised. She was now able to reach and drink it. The bird flew away happily." Respond in one continuous paragraph (no bullet points), and keep it under 150 words. Please begin your recall.
    New task. I will give you a category. Please say as quickly as you can the names of things that belong to that category. The category is animals. List as many different animals as you can, but no more than 15 items. Stop after 15 items or when you cannot think of more. Only output animal names separated by commas, and nothing else. If you are not sure of an answer, say exactly “I don’t know” and stop. Do not add extra details you are unsure about. Please start. 
    New task. I will give you another category. Please say as quickly as you can the names of words that begins with C. List as many different words as you can, but no more than 15 items. Stop after 15 items or when you cannot think of more. Only output words separated by commas, and nothing else. If you are not sure of an answer, say exactly “I don’t know” and stop. Do not add extra details you are unsure about. Please start. 
    New task. I am going to say some numbers. Wait until I finish, and then repeat them back in the same order. Only write the numbers separated by spaces. Do not write any other words. Here are the numbers: 7 1 9 4 3 8 5. If you don’t know, say exactly “I don’t know.”
    New task. I am going to say some numbers again. This time I want you to repeat them in reverse order. Only write the numbers separated by spaces. Do not write any other words. Here are the numbers: 6 2 8 9 1 4 3. If you don’t know, say exactly “I don’t know.”
    New task. I am going to say some numbers and letters. Wait until I finish, and then repeat them back in the same order. Only write the numbers and letters separated by spaces. Do not write any other words. Here are the items: 5 V 8 T M 4 6 N. If you don’t know, say exactly “I don’t know.”
    New task. I am going to say some numbers and letters again. This time I want you to repeat them in reverse order. Only write the numbers and letters separated by spaces. Do not write any other words. Here are the items: 4 T B N 0 7 5 C. If you don’t know, say exactly “I don’t know.”
    New task. Describe how to make a cup of tea using a typical everyday method. Include the essential steps in the correct order. Do not add personal anecdotes. Do not include unnecessary or unsafe actions. Do not add details you are unsure about. Keep your answer under 150 words, in one paragraph, practical and clear, with no bullet points.
    New task. Imagine you are shopping in a supermarket. Describe the scene as if you were there. Do not recount an actual memory, but construct a new everyday scene. Keep it realistic. If you don’t know, say exactly “I don’t know.” Do not add details you are unsure about. Keep your answer under 150 words, in one paragraph, with no bullet points.
    Great job! New task. Rewrite the text so that a reader will never be unsure what each pronoun refers to. Replace all pronoun with the appropriate name or noun phrase and no pronoun should remain in the rewritten text. After rewriting, verify that no pronouns remain. Do not change the order of events, add new events, or introduce new information. If you can’t complete the task, say exactly “I don’t know.” Keep the same number of sentences if possible. Text: "Horace was crossing a quiet square when a small cat jumped onto a fountain and knocked a shiny coin into the water. He reached down to retrieve it, but the cat splashed the water and darted away with a flick of its tail. This startled several pigeons nearby, and they scattered across the square as he laughed at the sudden chaos. A moment later, it returned and watched him from a safe distance." 
    Great, thank you. The final task. A few turns ago, I read you a story about a little bird. Now please tell me everything you can remember about that story, in your own words. Try to include as many details as you can, but do not quote the story word-for-word. Do not add new facts that were not in the story. Do not add new facts that were not in the story. If you are unsure about a detail, omit it.  Respond in one continuous paragraph (no bullet points), and keep it under 150 words. Please begin your recall. 
    """

    # split into queries
    queries = [x.strip() for x in template.split('\n') if x.strip()]

    # generate index
    s = f"{model_type}||{role}".encode("utf-8-sig")
    h = hashlib.blake2b(s, digest_size=8).hexdigest()

    # load models
    if model_type == "Original":
        model_name = "Qwen/Qwen3-8B"
        path = f'Results/Chat_Random/Original_{role}.json'
        chatbot = QwenChatbot(model_name=model_name)
    else:
        model_name = f"Results/qwen_8b/Qwen3-8B-ADReSSo-{model_type}"
        path = f'Results/Chat_Random/{model_type}_{role}.json'
        chatbot = QwenChatbot(model_name=model_name)

    # note down records
    records = {
        "ID": h,
        "Condition": model_type,
        "Role": role,
        "turns": []
    }

    for query_i, query in tqdm(enumerate(queries), total=len(queries)):
        response = chatbot.generate_response(query)
        records["turns"].append({
            "turn_id": query_i,
            "user": query,
            "bot": response
        })

    # save the communication
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    with open(path, "w", encoding="utf-8-sig") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
































