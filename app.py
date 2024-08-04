import streamlit as st
from llama_cpp import Llama

def create_prompt_with_chat_format(messages, bos="<s>", eos="</s>", add_bos=True):
    formatted_text = ""
    for message in messages:
        if message["role"] == "system":
            formatted_text += "<|system|>\n" + message["content"] + "\n"
        elif message["role"] == "user":
            formatted_text += "<|user|>\n" + message["content"] + "\n"
        elif message["role"] == "assistant":
            formatted_text += "<|assistant|>\n" + message["content"].strip() + eos + "\n"
        else:
            raise ValueError(
                "Tulu chat template only supports 'system', 'user' and 'assistant' roles. Invalid role: {}.".format(
                    message["role"]
                )
            )
    formatted_text += "<|assistant|>\n"
    formatted_text = bos + formatted_text if add_bos else formatted_text
    return formatted_text


def main():
    st.title("Chat Demo")

    llm = Llama(model_path="models/Airavata.gguf", n_ctx=4096, n_threads=2, n_gpu_layers=-1)

    generation_kwargs = {
        "max_tokens":4096,
        "stop":["</s>"],
        "echo":False,
        "top_k":50,
        "top_p":0.6
    }

    input_text = st.text_input("Enter your text", "")
    input_prompt = create_prompt_with_chat_format([{"role": "user", "content": input_text}], add_bos=False)
    output = llm(input_prompt, **generation_kwargs)

    if st.button("Enter"):
        output_text = output["choices"][0]["text"]
        st.write("Prediction:", output_text)

if __name__ == "__main__":
    main()
