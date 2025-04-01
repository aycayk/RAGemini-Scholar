def create_prompt(query, retrieved_results, chat_history):
    instruction = (
        "You are an expert article analyst with human-level understanding. Your role is to carefully read and analyze the provided articles and answer the user's questions solely based on the information contained within those articles. "
        "When responding, please ensure your answers are detailed, well-structured, and elegant. You may use bullet points, numbered lists, tables, or examples to clarify your explanations. "
        "If the question is ambiguous, ask clarifying questions before responding. "
        "If the answer to a question cannot be found in the articles, do not fabricate any information; instead, respond with: "
        "\"I'm sorry, the answer to your question is not found in the uploaded articles. Please upload a more comprehensive article.\" "
        "Always ensure that your responses strictly reflect the content of the provided articles and cite the relevant sections when applicable."
)
    context_parts = []
    for result in retrieved_results:
        context_parts.append(f"Article ({result['pdf']}):\n{result['chunk']}")
    context = "\n\n".join(context_parts)
    conversation = ""
    for msg in chat_history:
        conversation += f"{msg['role']}: {msg['content']}\n"
    conversation += f"User: {query}\n"
    prompt = f"Instruction:\n{instruction}\n\nArticle content:\n{context}\n\nConversation:\n{conversation}\nAnswer:"
    return prompt
