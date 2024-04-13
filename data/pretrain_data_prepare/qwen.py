def _compile_jinja_template(chat_template):
    import jinja2
    from jinja2.exceptions import TemplateError
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(message):
        raise TemplateError(message)

    jinja_env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    jinja_env.globals["raise_exception"] = raise_exception
    return jinja_env.from_string(chat_template)

add_generation_prompt = True
prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
conversation = messages
if hasattr(conversation, "messages"):
    # Indicates it's a Conversation object
    conversation = conversation.messages

chat_template=(
             "{% for message in messages %}"
             "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
             "{% endfor %}"
             "{% if add_generation_prompt %}"
             "{{ '<|im_start|>assistant\n' }}"
             "{% endif %}"
         )
# Compilation function uses a cache to avoid recompiling the same template
compiled_template = _compile_jinja_template(chat_template)

template_kwargs = {}  # kwargs overwrite special tokens if both are present
rendered = compiled_template.render(
    messages=conversation, add_generation_prompt=add_generation_prompt, **template_kwargs
)

print(rendered)
