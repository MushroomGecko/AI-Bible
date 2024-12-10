import json

import embedding
import milvuslitebible

dbname = 'milvuslitebible'
cname = 'milvuslitebible_nasb1995'

client = milvuslitebible.get_database(dbname)
if not client.list_collections():
    with open("NASB1995_bible.json", "r", encoding='utf-8-sig') as file:
        bible_json = json.load(file)
    id_count = 0
    for book in bible_json:
        for chapter in bible_json[book]:
            titles = []
            texts = []
            ids = []
            embeddings = None
            for verse in bible_json[book][chapter]:
                titles.append(f'{book} {chapter}:{verse}')
                texts.append(bible_json[book][chapter][verse])
                ids.append(id_count)
                id_count += 1
            embeddings = embedding.get_embedding(text=texts, mode='sentence')
            if not client.list_collections():
                client = milvuslitebible.create_collection(collection_name=cname, database_name=dbname, embeddings=embeddings, metric='L2')
                print(f'Collection {cname} does not exist. Created collection {cname}.')
            milvuslitebible.insert_data(collection_name=cname, client=client, embeddings=embeddings, texts=texts, titles=titles, ids=ids)
            print(f'Inserted {book} {chapter}')
else:
    print(client.list_collections())
    client = milvuslitebible.get_database(dbname)
    print(milvuslitebible.search_collection(query='In the beginning God created the heavens and the earth.', client=client, collection_name=cname, metric='L2'))
    client.close()
