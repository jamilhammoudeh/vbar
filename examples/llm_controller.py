"""Fly one approach episode with a language-model controller.

Fill in complete() with a call to your model client. Until a key is set the run uses a scripted
stand-in and needs no network. Run it with: python examples/llm_controller.py

The key and model name are read from VBAR_LLM_API_KEY and VBAR_LLM_MODEL. Copy .env.example to
.env to keep them in a file; that needs python-dotenv, installed by the llm extra.
"""

import os

from vbar import ApproachConfig, ApproachEnv, run_episode
from vbar.agents import LLMAgent, Reply


def settings():
    """The API key and model name, each None when unset or empty.

    With python-dotenv installed, a .env file in the working folder or a parent is read first; a
    variable that is already set is kept.
    """
    try:
        import dotenv
    except ImportError:
        pass
    else:
        dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))
    return os.environ.get("VBAR_LLM_API_KEY") or None, os.environ.get("VBAR_LLM_MODEL") or None


def complete(messages, tools):
    """Send the conversation to your model and return its reply.

    Build the request from `messages`, using the key and model name from settings(), and return
    the reply text, or a Reply with the token counts the provider reports. `tools` is empty
    unless the agent is given tools.
    """
    raise NotImplementedError("fill in complete() with a call to your model client")


def stand_in(messages, tools):
    """A scripted reply that commands zero thrust."""
    return Reply('{"ax": 0.0, "ay": 0.0, "az": 0.0}')


def main():
    key, model = settings()
    if key:
        agent = LLMAgent(complete, name=model or "llm")
    else:
        print("VBAR_LLM_API_KEY is not set: flying the scripted stand-in, which commands no thrust")
        agent = LLMAgent(stand_in, name="stand-in")
    env = ApproachEnv(ApproachConfig(start_state=(0, -400, 0, 0, 0, 0)))
    result = run_episode(env, agent)
    print(
        f"{result['time_s']:.0f} s, {result['final_dist_to_hold_m']:.0f} m from the hold point, "
        f"delta-v {result['dv_used_mps']:.2f} m/s, keep-out violated: {result['keep_out_violated']}"
    )
    print(agent.usage())


if __name__ == "__main__":
    main()
