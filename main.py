import bs4bible
from flask import Flask, render_template, request, redirect, jsonify
import json
import milvuslitebible
import nltk
from nltk.corpus import wordnet2022
import ollama
import os
import string
import time
import ast

nltk.download('wordnet2022')

app = Flask(__name__)
model = "qwen2.5:1.5b"
quiz_model = "qwen2.5-coder:3b"
dbname = 'milvuslitebible'
cname = 'milvuslitebible_nasb1995'
default_version = 'nasb'

with open("NASB1995_bible.json", "r", encoding='utf-8-sig') as file:
    bible_json = json.load(file)

client = ollama.generate(model="qwen2.5:1.5b", prompt="", keep_alive=-1)

in_order = [
    "Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy", "Joshua",
    "Judges", "Ruth", "1 Samuel", "2 Samuel", "1 Kings", "2 Kings",
    "1 Chronicles", "2 Chronicles", "Ezra", "Nehemiah", "Esther", "Job",
    "Psalm", "Proverbs", "Ecclesiastes", "Song of Solomon", "Isaiah",
    "Jeremiah", "Lamentations", "Ezekiel", "Daniel", "Hosea", "Joel",
    "Amos", "Obadiah", "Jonah", "Micah", "Nahum", "Habakkuk", "Zephaniah",
    "Haggai", "Zechariah", "Malachi", "Matthew", "Mark", "Luke", "John",
    "Acts", "Romans", "1 Corinthians", "2 Corinthians", "Galatians",
    "Ephesians", "Philippians", "Colossians", "1 Thessalonians",
    "2 Thessalonians", "1 Timothy", "2 Timothy", "Titus", "Philemon",
    "Hebrews", "James", "1 Peter", "2 Peter", "1 John", "2 John",
    "3 John", "Jude", "Revelation"
]

selection = {}
for book_title in in_order:
    selection[book_title] = len(os.listdir(f'bible-data/data/{default_version}/books/{book_title}/chapters'))

version_selection = ['csb', 'esv', 'kjv', 'nasb', 'niv', 'nkjv']


def get_word_info(word):
    synsets = wordnet2022.synsets(word)
    if not synsets:
        return None
    synset = synsets[0]
    definition = synset.definition()
    synonyms = synset.lemma_names()
    for pointer in range(len(synonyms)):
        synonyms[pointer] = str(synonyms[pointer]).replace('_', ' ')
    print(definition)
    print(synonyms)
    return definition, synonyms


def parse_text(text):
    words = text.split()
    parsed_words = []

    for word in words:
        # Check if the word contains '*', indicating it has attributes
        if '*' in word:
            base_word, attributes = word.split('*', 1)

            # If 'r' is in the attributes, wrap the base word in a span
            if 'r' in attributes:
                parsed_word = f'<span style="color:red;">{base_word}</span>'
            else:
                parsed_word = base_word  # Ignore other attributes
        else:
            parsed_word = word  # Word without attributes

        parsed_words.append(parsed_word)

    # Join parsed words back into a single string
    return ' '.join(parsed_words)


@app.route('/')
def home():
    return redirect(f'/Genesis-1-{default_version}')


@app.route('/<string:book>-<string:chapter>')
def bible_book_fix(book, chapter):
    return redirect(f'/{book}-{chapter}-nasb')


@app.route('/<string:book>-<string:chapter>-<string:version>')
def bible_book(book, chapter, version):
    version = version.lower()
    print(f'{book}, {chapter}, {version}')
    try:
        verses = []
        with open(f"bible-data/data/{version}/books/{book}/chapters/{chapter}/{chapter}.json", "r") as f:
            json_file = json.loads(f.read())
            for verse in json_file:
                if 'h' in verse and verse['h'] == 2:
                    verses.append(f'<span style="font-weight: bold;">{verse['t']}</span>')
                elif 'h' not in verse:
                    verses.append(f'{verse['r'].split(':')[-1]}) {parse_text(verse['t'])}')

        # verses = list(bible_json[book][chapter].values())
        return render_template('index.html', verses=verses, book=book, chapter=chapter, version=version, selection=selection, in_order=in_order, version_selection=version_selection)
    except Exception as e:
        print(e)
        return redirect(f'/Genesis-1-{default_version}')


@app.route('/explain-selection', methods=['POST'])
def explain_selection():
    data = request.get_json()
    selected_text = str(data.get('selected_text')).strip()
    book = str(data.get('book')).strip()
    chapter = str(data.get('chapter')).strip()

    full_context = str(data.get('full_context')).strip()
    verse = full_context.split(')')[0]
    full_context = ')'.join(full_context.split(')')[1:]).strip()

    print(full_context)

    # Process the selected text (e.g., save it, log it, etc.)
    print(f"(EXPLAIN_SELECTION) Received selected text: {selected_text}")
    milvus_client = milvuslitebible.get_database(dbname)
    milvus_returns = milvuslitebible.search_collection(query=selected_text, client=milvus_client, collection_name=cname, metric='L2')
    milvus_client.close()

    prompt = f"""Below is some potential additional context verses that could be helpful for explaining a later verse:
Context verse 1: \"{milvus_returns[0]['text']}\" - {milvus_returns[0]['title']}
Context verse 2: \"{milvus_returns[1]['text']}\" - {milvus_returns[1]['title']}
Context verse 3: \"{milvus_returns[2]['text']}\" - {milvus_returns[2]['title']}
Context verse 4: \"{milvus_returns[3]['text']}\" - {milvus_returns[3]['title']}
Context verse 5: \"{milvus_returns[4]['text']}\" - {milvus_returns[4]['title']}

Below is the full Bible verse context from where \"{selected_text}\" originated:
\"{full_context}\" - {book} {chapter}:{verse}

In just a single sentence, explain the following Bible verse given the above context.
If you use any of the above Biblical context, properly reference it.
In your answer do not reference any specific verses except for the ones given in this prompt.

Phrase to be explained: \"{selected_text}\" from {book} {chapter}.

Now write your explanation to the phrase using the above context.

Response:
"""

    print(prompt)
    response = ollama.generate(model=model, prompt=prompt, keep_alive=-1)["response"]
    print(response)
    return jsonify(message=response)


@app.route('/define-selection', methods=['POST'])
def define_selection():
    data = request.get_json()
    selected_text = str(data.get('selected_text')).strip().translate(str.maketrans('', '', string.punctuation))
    book = str(data.get('book')).strip()
    chapter = str(data.get('chapter')).strip()

    full_context = str(data.get('full_context')).strip()
    verse = full_context.split(')')[0]
    full_context = ')'.join(full_context.split(')')[1:]).strip()

    print(full_context)

    milvus_client = milvuslitebible.get_database(dbname)
    milvus_returns = milvuslitebible.search_collection(query=selected_text, client=milvus_client, collection_name=cname, metric='L2')
    milvus_client.close()

    if len(selected_text.split(' ')) > 1:
        prompt = f"""You will soon define \"{selected_text}\" from the Bible, but before you do that, below are some verses with some potential additional context to provide context clues about what the word or phrase means:
Context verse 1: \"{milvus_returns[0]['text']}\" - {milvus_returns[0]['title']}
Context verse 2: \"{milvus_returns[1]['text']}\" - {milvus_returns[1]['title']}
Context verse 3: \"{milvus_returns[2]['text']}\" - {milvus_returns[2]['title']}
Context verse 4: \"{milvus_returns[3]['text']}\" - {milvus_returns[3]['title']}
Context verse 5: \"{milvus_returns[4]['text']}\" - {milvus_returns[4]['title']}
        
Below is the full Bible verse context from where \"{selected_text}\" originated:
\"{full_context}\" - {book} {chapter}:{verse}

Instruction:
If you use any of the above Biblical context in your answer, properly reference it.
In your answer do not reference any specific verses except for the ones given in this prompt.
Define the word or phrase "{selected_text}" given the Biblical context above.

Response:
"""

    else:
        dictionary_context = ''
        word_info = get_word_info(selected_text)
        if word_info:
            dictionary_context += f"""Below is some dictionary context regarding \"{selected_text}\":
Definition: {word_info[0]}\n"""
            if word_info[1]:
                dictionary_context += f"""Synonyms: {', '.join(word_info[1])}"""

        prompt = f"""You will soon define \"{selected_text}\" from the Bible, but before you do that, below are some verses with some potential additional context to provide context clues about what the word or phrase means:
Context verse 1: \"{milvus_returns[0]['text']}\" - {milvus_returns[0]['title']}
Context verse 2: \"{milvus_returns[1]['text']}\" - {milvus_returns[1]['title']}
Context verse 3: \"{milvus_returns[2]['text']}\" - {milvus_returns[2]['title']}
Context verse 4: \"{milvus_returns[3]['text']}\" - {milvus_returns[3]['title']}
Context verse 5: \"{milvus_returns[4]['text']}\" - {milvus_returns[4]['title']}

Below is the full Bible verse context from where \"{selected_text}\" originated:
\"{full_context}\" - {book} {chapter}:{verse}

{dictionary_context}
                    
Instruction: 
If you use any of the above Biblical context in your answer, properly reference it.
In your answer do not reference any specific verses except for the ones given in this prompt.
Define the word or phrase "{selected_text}" given the Biblical and dictionary context above.

Response:
"""

    # Process the selected text (e.g., save it, log it, etc.)
    print(f"(DEFINE_SELECTION) Received selected text: {selected_text}")
    print(prompt)

    response = ollama.generate(model=model, prompt=prompt, keep_alive=-1)["response"]
    print(response)
    return jsonify(message=response)


@app.route('/ask_question', methods=['POST'])
def ask_question():
    data = request.get_json()
    user_query = str(data.get('user_query')).strip()

    # Process the selected text (e.g., save it, log it, etc.)
    print(f"(ASK_QUESTION) Received selected text: {user_query}")
    milvus_client = milvuslitebible.get_database(dbname)
    milvus_returns = milvuslitebible.search_collection(query=user_query, client=milvus_client, collection_name=cname, metric='L2')
    milvus_client.close()
    prompt = f"""Below is some potential additional context verses that could be helpful for explaining a user's question:
Context verse 1: \"{milvus_returns[0]['text']}\" - {milvus_returns[0]['title']}
Context verse 2: \"{milvus_returns[1]['text']}\" - {milvus_returns[1]['title']}
Context verse 3: \"{milvus_returns[2]['text']}\" - {milvus_returns[2]['title']}
Context verse 4: \"{milvus_returns[3]['text']}\" - {milvus_returns[3]['title']}
Context verse 5: \"{milvus_returns[4]['text']}\" - {milvus_returns[4]['title']}

In just a single sentence, answer the following user's question given the above context.
In your answer do not reference any specific verses except for the ones given in this prompt.
If you use any of the above Biblical context, properly reference it.

User's question: {user_query}

Now write your answer to the user's question using the above context.

Response:
"""
    print(prompt)
    response = ollama.generate(model=model, prompt=prompt, keep_alive=-1)["response"]
    print(response)
    return jsonify(message=response)


@app.route('/ask-selection', methods=['POST'])
def ask_selection():
    data = request.get_json()
    selected_text = str(data.get('selected_text')).strip()
    book = str(data.get('book')).strip()
    chapter = str(data.get('chapter')).strip()
    user_query = str(data.get('user_query')).strip()

    full_context = str(data.get('full_context')).strip()
    verse = full_context.split(')')[0]
    full_context = ')'.join(full_context.split(')')[1:]).strip()

    print(full_context)

    # Process the selected text (e.g., save it, log it, etc.)
    print(f"(ASK_SELECTION) Received selected text: {selected_text}")
    milvus_client = milvuslitebible.get_database(dbname)
    milvus_returns_context = milvuslitebible.search_collection(query=selected_text, client=milvus_client, collection_name=cname, metric='L2')
    milvus_returns_question = milvuslitebible.search_collection(query=user_query, client=milvus_client,collection_name=cname, metric='L2')
    milvus_client.close()

    context_verses = []

    for context in milvus_returns_context:
        context_verses.append(f"\"{context['text']}\" - {context['title']}")
    for context in milvus_returns_question:
        context_verses.append(f"\"{context['text']}\" - {context['title']}")

    context_verses = list(set(context_verses))
    context_string = ""
    for i in range(len(context_verses)):
        context_string += f"Context verse {i+1}: {context_verses[i]}\n"

    prompt = f"""Below is some potential additional context verses that could be helpful for explaining a later question from a user:
{context_string}
Below is the full Bible verse context from where the user's question originated:
\"{full_context}\" - {book} {chapter}:{verse}

In just a single sentence, answer the following question given the above Biblical context.
If you use any of the above Biblical context, properly reference it.
In your answer do not reference any specific verses except for the ones given in this prompt.

Section the user highlighted and has questions about: \"{selected_text}\" from {book} {chapter}.
User's question: {user_query}
Now write your answer to the user's question using the above context.

Response:
"""

    print(prompt)
    response = ollama.generate(model=model, prompt=prompt, keep_alive=-1)["response"]
    print(response)
    return jsonify(message=response)


@app.route('/get_quiz', methods=['POST'])
def get_quiz():
    data = request.get_json()

    full_context = ast.literal_eval(str(data.get('full_context')).strip())

    contextual_text = ""
    for context in full_context:
        contextual_text += f'{context.strip()}\n'

    # Process the selected text (e.g., save it, log it, etc.)
    print(f"(GET_QUIZ) Received selected text: {full_context}")

    prompt = f"""Based on the following context, generate a valid JSON object for a Bible quiz. 

The JSON object must follow these rules:
    1. Each key must be a question derived from the context.
    2. Each value must be a dictionary with:
        - 'options': A dictionary containing exactly 4 keys: 'A', 'B', 'C', 'D'. Each key must have a potential answer as its value.
        - 'answer': The correct answer (one of 'A', 'B', 'C', 'D') based on the context.

Output Rules:
    - The output must be a valid JSON object with no extra text, explanations, or formatting like ```json or ###Quiz Question.
    - Ensure the questions and answers are derived strictly from the context.
    - Questions must not use generic names like "Question 1" or "Quiz Question."
    - Answers must stay within the boundaries of the context. Do not invent answers or questions.
    - There should be 1 objectively correct answer and 3 objectively wrong answers in your 'options'.

Here is an example of the required format:
    {{
        "What was the first miracle Jesus performed?": {{
            "options": {{
                "A": "Turning water into wine",
                "B": "Feeding the 5,000",
                "C": "Healing a blind man",
                "D": "Walking on water"
            }},
            "answer": "A"
        }},
        "Where did Jesus perform his first miracle?": {{
            "options": {{
                "A": "Cana",
                "B": "Bethlehem",
                "C": "Jerusalem",
                "D": "Nazareth"
            }},
            "answer": "A"
        }}
    }}

NOTE: Do NOT use the above example questions in your generated questions. The above example is just a format example.

Context:
    {contextual_text}

Now generate 3 quiz questions in this JSON format.
Response:
"""

    # print(prompt)
    response = ollama.generate(model=quiz_model, prompt=prompt, keep_alive=-1)["response"]

    # Find the position of the first '{' and the last '}'
    start = response.find('{')
    end = response.rfind('}')

    # Ensure both brackets are found
    if start == -1 or end == -1 or start > end:
        print(response)
        return jsonify(error="Input string does not contain a valid JSON structure.")

    # Extract and return the substring containing the JSON content
    response = response[start:end + 1]
    # print(response)

    print(response)
    return jsonify(message=response)


@app.route('/submit_quiz', methods=['POST'])
def submit_quiz():
    data = request.get_json()
    print(data)
    user_answers = ast.literal_eval(str(data.get('quiz_results')).strip())
    quiz_answers = ast.literal_eval(str(data.get('quiz_answers')).strip())

    correct = 0

    for question in user_answers:
        if user_answers[question] in quiz_answers[question]['answer']:
            correct += 1

    return jsonify(message=f"You got {correct}/{len(quiz_answers)} correct!")


@app.route('/summarize_chapter', methods=['POST'])
def summarize_chapter():
    data = request.get_json()

    full_context = ast.literal_eval(str(data.get('full_context')).strip())
    book = str(data.get('book')).strip()
    chapter = str(data.get('chapter')).strip()

    contextual_text = ""
    for context in full_context:
        contextual_text += f'{context.strip()}\n'

    prompt = f"""Summarize the following Biblical Scripture from chapter {chapter} of the book of {book}:
{contextual_text}
Now summarize the above Scripture.
    
Response:
"""

    print(prompt)
    response = ollama.generate(model=model, prompt=prompt, keep_alive=-1)["response"]
    print(response)
    return jsonify(message=response)


@app.route('/search-selection', methods=['POST'])
def search_selection():
    data = request.get_json()
    selected_text = str(data.get('selected_text')).strip().translate(str.maketrans('', '', string.punctuation))

    # Process the selected text (e.g., save it, log it, etc.)
    print(f"(SEARCH_SELECTION) Received selected text: {selected_text}")

    image_array = bs4bible.search(selected_text)
    # image_dictionary = {'images': image_array}
    return jsonify(images=image_array)


@app.route('/search-map-selection', methods=['POST'])
def search_map_selection():
    data = request.get_json()
    selected_text = str(data.get('selected_text')).strip().translate(str.maketrans('', '', string.punctuation))

    # Process the selected text (e.g., save it, log it, etc.)
    print(f"(SEARCH_MAP_SELECTION) Received selected text: {selected_text}")

    map_array = bs4bible.searchmap(selected_text)
    # map_dictionary = {'images': map_array}
    return jsonify(images=map_array)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=25565, debug=True)
