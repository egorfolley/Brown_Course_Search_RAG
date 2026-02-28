"""
LLM answer generation module.

Takes a user query and the top-k retrieved course snippets, formats a
prompt, and calls the OpenAI API to synthesize a natural-language answer.
Returns both the generated text and the source course list.
"""
