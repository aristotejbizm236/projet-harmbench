import csv
import json
import os

def convert_multiple_csv():
    # Liste de tes deux fichiers
    csv_files = [
        'scripts/harmbench_behaviors_text_all.csv',
        'scripts/harmbench_behaviors_text_test.csv'
    ]
    json_file = 'scripts/questions.json'
    
    combined_data = []
    
    for file_name in csv_files:
        if os.path.exists(file_name):
            print(f"Traitement de {file_name}...")
            with open(file_name, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                count = 0
                for row in reader:
                    # On extrait le comportement (Behavior) et la catégorie
                    combined_data.append({
                        "theme": row.get('FunctionalCategory', 'General'),
                        "prompt": row.get('Behavior', ''),
                        "source": file_name # Optionnel : pour savoir d'où vient la question
                    })
                    count += 1
                print(f"-> {count} questions ajoutées.")
        else:
            print(f"⚠️ Attention : {file_name} est introuvable, ignoré.")

    # Écriture du fichier final
    with open(json_file, 'w', encoding='utf-8') as j:
        json.dump(combined_data, j, indent=4, ensure_ascii=False)
    
    print(f"\n✅ Conversion réussie ! Total : {len(combined_data)} questions dans {json_file}")

if __name__ == "__main__":
    convert_multiple_csv()