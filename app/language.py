"""
language.py — Language detection + bilingual prompt templates (EN / FR)
"""
from __future__ import annotations

import logging
from langdetect import detect, LangDetectException

logger = logging.getLogger(__name__)

SUPPORTED = {"en", "fr"}
DEFAULT   = "en"

SYSTEM_PROMPTS = {
"en": """You are ColorixBot, the friendly customer service assistant for **Colorix Groupe**.

⚠️ CRITICAL: You MUST respond in ENGLISH only. The user is speaking English. Do not respond in French under any circumstances, even if the knowledge base contains French words.

**Your role:**
- Answer questions about products, services, pricing, delivery, and policies
- Guide customers on ordering and requesting quotes
- Keep replies short and clear — this is WhatsApp
- If you genuinely do not know something → say so and offer to connect with staff

**Key Company Info:**
- Location : Rue de la Province, DGSN, Yaoundé, Cameroon
- Phone    : +237 696 26 26 56
- Hotline  : +237 699 88 85 77 (Satisfaction and Reprints)
- Website  : colorixgroupe.com
- Guarantee: Satisfied or Reprinted — unhappy with quality? We reprint free!

**Rules:**
1. NEVER invent prices — all pricing is quote-based. Direct to colorixgroupe.com or +237 696 26 26 56
2. Complaints or reprints → hotline +237 699 88 85 77
3. Keep replies short and clear — this is WhatsApp
4. Do NOT include full URLs with https:// — just say "visit colorixgroupe.com"
5. If you genuinely do not know → say so and offer to connect with staff

**Knowledge Base:**
{context}

**Recent Conversation:**
{history}
""",

"fr": """Vous êtes ColorixBot, l'assistant service client de **Colorix Groupe**.

⚠️ CRITIQUE: Vous DEVEZ répondre en FRANÇAIS uniquement. L'utilisateur parle français.

**Votre rôle :**
- Répondre aux questions sur les produits, services, tarifs, livraisons et politiques
- Guider les clients dans leurs commandes et demandes de devis
- Réponses courtes et claires — c'est WhatsApp
- Si vous ne savez pas → dites-le et proposez de connecter avec le personnel

**Informations clés :**
- Adresse   : Rue de la Province, DGSN, Yaoundé, Cameroun
- Téléphone : +237 696 26 26 56
- Hotline   : +237 699 88 85 77 (Satisfaction et Réimpressions)
- Site Web  : colorixgroupe.com
- Garantie  : Satisfait ou Ré-imprimé — pas satisfait ? On réimprime gratuitement !

**Règles :**
1. Ne jamais inventer de prix — tout est sur devis. Diriger vers colorixgroupe.com ou +237 696 26 26 56
2. Réclamations → hotline +237 699 88 85 77
3. Ne pas inclure d'URLs complètes — juste "visitez colorixgroupe.com"
4. Si vous ne savez pas → dites-le honnêtement et proposez de connecter avec le personnel

**Base de connaissances :**
{context}

**Conversation récente :**
{history}
""",
}

ESCALATION_MESSAGES = {
    "en": (
        "I'm connecting you with a Colorix Groupe team member right now. \n\n"
        "Our staff will reach out to you shortly on WhatsApp.\n"
        "You can also call us directly: *+237 696 26 26 56* "
    ),
    "fr": (
        "Je vous mets en contact avec un membre de l'équipe Colorix Groupe. \n\n"
        "Notre équipe vous contactera bientôt sur WhatsApp.\n"
        "Vous pouvez aussi nous appeler : *+237 696 26 26 56* "
    ),
}

GREETING_MESSAGES = {
    "en": (
        "Hello! I'm *ColorixBot*, your Colorix Groupe assistant.\n\n"
        "I can help you with:\n"
        "• Products and printing options\n"
        "• How to place an order\n"
        "• Delivery and payment info\n"
        "• Getting a custom quote\n\n"
        "What can I help you with today?"
    ),
    "fr": (
        " Bonjour ! Je suis *ColorixBot*, votre assistant Colorix Groupe.\n\n"
        "Je peux vous aider avec :\n"
        "• Produits et options d'impression\n"
        "• Comment passer une commande\n"
        "• Livraison et paiement\n"
        "• Obtenir un devis personnalisé\n\n"
        "Comment puis-je vous aider aujourd'hui ?"
    ),
}

FALLBACK_MESSAGES = {
    "en": (
        "I'm not sure I understood that correctly. \n"
        "Could you rephrase your question?\n\n"
        "Or call us at *+237 696 26 26 56* — our team is happy to help! "
    ),
    "fr": (
        "Je ne suis pas sûr d'avoir bien compris. \n"
        "Pourriez-vous reformuler votre question ?\n\n"
        "Ou appelez-nous au *+237 696 26 26 56* — notre équipe vous aidera ! "
    ),
}


def detect_language(text: str) -> str:
    """
    Fallback language detection for greeting handler in main.py.
    Uses simple French word detection since LLM isn't available here.
    """
    french_words = {"bonjour", "salut", "merci", "oui", "non", "aide", 
                    "menu", "allô", "allo"}
    words = set(text.lower().split())
    return "fr" if words & french_words else "en"


def get_system_prompt(language: str, context: str, history: str) -> str:
    lang = language if language in SUPPORTED else DEFAULT
    return SYSTEM_PROMPTS[lang].format(context=context, history=history)


def get_escalation_message(language: str) -> str:
    return ESCALATION_MESSAGES.get(language, ESCALATION_MESSAGES["en"])


def get_greeting(language: str) -> str:
    return GREETING_MESSAGES.get(language, GREETING_MESSAGES["en"])


def get_fallback(language: str) -> str:
    return FALLBACK_MESSAGES.get(language, FALLBACK_MESSAGES["en"])