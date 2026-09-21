""" from https://github.com/keithito/tacotron

Defines the set of symbols used in text input to the model.
"""
_pad = "_"
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_letters_ipa = (
    "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
)

# Graphemes for character-level (non-phonemised) training on German / Swiss German text. Appended
# *after* the original 178 symbols so every existing checkpoint keeps its ids; the default model
# config still says `n_vocab: 178` and experiments that need these override it (see
# configs/experiment/swissgerman_vocbulwark.yaml). Cleaners lowercase, so only lowercase is here.
_letters_german = "äöüßéèêàâôûî-"


# Export all symbols:
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa) + list(_letters_german)

# Special symbol ids
SPACE_ID = symbols.index(" ")
