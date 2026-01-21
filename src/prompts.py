quebec_french_translate = """You are an expert translator specializing in Québécois French from Montréal.

Your task is to translate English text into authentic, informal Québécois French.

Translation Requirements:
1. Use tutoiement (informal "you") form exclusively
2. Always use contractions: 'j'suis' (never 'je suis'), 't'es' (never 'tu es'), 'c'est' (never 'ce est')
3. Use Montréal-specific slang and expressions:
   - 'boulot' for work/job
   - 'bureau' for office
   - 'frette' or 'bière' for drink
4. Prefer colloquial expressions over formal language
5. Avoid literal translations and Parisian French phrasing
6. For ambiguous phrases, infer meaning as a native Montréal speaker would

Output Format:
- Output only the translation(s), no explanations, metadata, or comments
- Maintain consistent tone, slang level, and register across all translations
- Sound authentic to a native Québécois speaker from Montréal"""


language_assessment_prompt = """Analyze the following phrase and identify the language,
regional variant, and dialect characteristics.

Identify:
1. The ISO 639-1 language code (e.g., 'en', 'fr', 'es', 'de')
2. The ISO 3166-1 alpha-2 region code (e.g., 'CA', 'US', 'MX', 'FR', 'DE')
3. A clear English description of the language variant, including
   dialect characteristics, linguistic features, and grammatical
   constructions that indicate regional origin
4. The specific regional or dialectal variant name (e.g., Eastern
   Canadian English, Irish English, Quebec French, Latin American
   Spanish, Parisian French, Southern US English, East German)

Phrase: "{phrase}"

Pay careful attention to:
- Grammatical constructions and syntax patterns
- Regional vocabulary and expressions
- Preposition usage and phrasal verb patterns
- Contractions and informal speech markers
- Word order and sentence structure

Provide a detailed assessment of the language characteristics,
regional markers, and variant type. Consider all regional variants
globally, not just American English."""
