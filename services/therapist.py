import os
from openai import OpenAI

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_GROQ_API_KEY = "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"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"

def generate_fallback_tips(target_text: str, user_transcription: str, mismatched_items: list) -> str:
    """
    Generate structured, friendly speech therapy advice when LLM is unavailable.
    """
    if not mismatched_items or target_text.strip().lower() == user_transcription.strip().lower():
        return (
            "🎉 **Outstanding Job!** Your pronunciation was crisp, accurate, and confident! "
            "Keep practicing with more phrases to build even greater fluency."
        )

    tips = [
        f"👋 **Hi there! I'm Speaky, your speech coach.** Let's work together to polish your pronunciation of **'{target_text}'**.\n",
        f"🔍 **What we noticed:** You said *'{user_transcription}'*. Let's focus on the sounds: **{', '.join(mismatched_items)}**.\n",
        "💡 **Step-by-Step Practice Guide:**"
    ]

    for item in mismatched_items[:3]:
        tips.append(
            f"- **For the sound '{item}':**\n"
            f"  1. Place your tongue gently and feel where the airflow goes.\n"
            f"  2. Say the sound slowly: *'{item}... {item}... {item}'*.\n"
            f"  3. Blend it back into the word: *'{target_text}'*."
        )

    tips.append(
        "\n🌟 **Fun Practice Tip:** Practice in front of a mirror to watch your lip and tongue shape! "
        "Take a deep breath and give it another try—you're doing fantastic! 🚀"
    )

    return "\n".join(tips)

def get_speech_therapy_tips(target_text: str, user_transcription: str, mismatched_items: list) -> str:
    """
    Generate personalized speech therapy advice using Groq LLM,
    with automatic fallback to curated rules.
    """
    if not mismatched_items or target_text.strip().lower() == user_transcription.strip().lower():
        return (
            "🎉 **Awesome job!** Your pronunciation matches the target accurately! "
            "Keep up the brilliant effort!"
        )

    mismatched_str = ", ".join(mismatched_items)
    api_key = os.getenv("GROQ_API_KEY", DEFAULT_GROQ_API_KEY)
    model_name = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)

    try:
        client = OpenAI(
            base_url=GROQ_BASE_URL,
            api_key=api_key,
            timeout=8.0
        )

        prompt = f"""You are "Speaky", a kind, warm, and encouraging speech therapist working with young learners and speech students.
The user was asked to pronounce the target phrase: "{target_text}"
The user spoke and was transcribed as: "{user_transcription}"
The sounds or words needing work are: {mismatched_str}

Please provide a structured, motivating response with:
1. A warm, encouraging 1-sentence opening.
2. Clear mouth and tongue movement instructions for the challenging sound(s).
3. A slow syllable breakdown (e.g. sound-by-sound).
4. A quick 10-second practice exercise and an encouraging closing message.
Keep your response concise, well-formatted with markdown, and very supportive."""

        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a caring, encouraging speech therapist named Speaky."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=450
        )

        content = completion.choices[0].message.content
        if content and len(content.strip()) > 30:
            return content.strip()

    except Exception as e:
        print(f"[Notice] Groq LLM API call failed or timed out ({e}). Using rich fallback advice.")

    return generate_fallback_tips(target_text, user_transcription, mismatched_items)
