from models.tisasrec import TiSASRec
import torch
from pathlib import Path
import json
from pprint import pprint

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

CATALOG = {
    "Mobile Apps": {
        "Productivity": ["Notion", "Evernote", "Slack", "Todoist"],
        "Social Media": ["Instagram", "TikTok", "X (Twitter)", "Threads"],
        "Finance": ["Mint", "Robinhood", "Revolut", "YBAB"]
    },
    "Games": {
        "RPG": ["Genshin Impact", "Elden Ring", "Final Fantasy VII", "Starfield"],
        "Action": ["Call of Duty", "Hades", "Doom Eternal", "Spider-Man 2"],
        "Puzzle": ["Portal 2", "Tetris Effect", "Monument Valley", "Candy Crush"]
    }
}

MAIN_CATEGORIES = ["Mobile Apps", "Games"]

def clear_screen():
    print("\n" * 2)

def get_choice(options):
    """Helper to handle numbered menu selection."""
    for i, option in enumerate(options, 1):
        print(f"{i}. {option}")

    while True:
        try:
            choice = int(input("\nSelect an option (number): "))
            if 1 <= choice <= len(options):
                return options[choice - 1]
            print(f"Please pick a number between 1 and {len(options)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def main():
    selected_items = []
    print("--- Welcome to the App & Game Discovery Tool ---")

    # Stage 1: Initial Category Selection
    print("\nWhat are you interested in today?")
    main_category = get_choice(MAIN_CATEGORIES)

    #* default to mobilerec settings
    settings = {
        "num_items": 10173,
        "num_categories": 48,
    }
    model_checkpoint = Path("metadata") / "mobilerec_model.pt"

    #* switch to steamrec settings
    if main_category == MAIN_CATEGORIES[1]:
        settings = {
            "num_items": 9597,
            "num_categories": 17,
        }
        model_checkpoint = Path("metadata") / "steamrec_model.pt"
        with open('metadata/game_categories.json', 'r') as file:
            categories = json.load(file)
        with open('metadata/game_metadata.json', 'r') as file:
            metadata = json.load(file)
    else:
        with open('metadata/app_categories.json', 'r') as file:
            categories = json.load(file)
        with open('metadata/app_metadata.json', 'r') as file:
            metadata = json.load(file)

    model = TiSASRec(
        num_items=settings["num_items"],
        num_categories=settings["num_categories"],
        num_metadata=5,
        max_len=50,
        time_span=256,
        hidden_size=128,
        num_blocks=2,
        num_heads=2,
        dropout=0.2,
        device=device,
    )

    model.load_state_dict(torch.load(model_checkpoint, map_location=device))

    while True:
        # Stage 2: Show subcategories based on Stage 1
        print(f"\n--- {main_category} Categories ---")
        subcategories = list(categories.keys())
        selected_subcategory = get_choice(subcategories)

        # Stage 3: Show Items in subcategory
        print(f"\n--- Category: {selected_subcategory} ({main_category}) ---")
        item_ids = categories[selected_subcategory]["item_ids"]
        item_names = [metadata.get(item_id) for item_id in item_ids]
        item_choice = get_choice(item_names)

        if item_choice not in selected_items:
            selected_items.append(item_choice)
            print(f"Added '{item_choice}' to your list!")
        else:
            print(f"'{item_choice}' is already in your list.")

        # Stage 4 & 5: Loop or Finish
        print("\nWould you like to add more?")
        continue_choice = get_choice([f"Yes, add more {main_category.lower()}", "No, I'm finished"])

        if continue_choice == "No, I'm finished":
            break
        clear_screen()

    scores = model.score_all_items(
        input_ids=input_ids,
        metadata_seq=metadata_seq,
        category_seq=category_seq,
        time_matrix=time_matrix,
    )

    pprint(scores)

    # Stage 6: Final Recommendations Output
    print("\n" + "="*60)
    print(f"Top 10 {main_category} Recommendations (based on your selection)")
    print("="*60)
    if not selected_items:
        print("No items selected.")
    else:
        for idx, item in enumerate(selected_items, 1):
            print(f"{idx}. {item}")
    print("="*60)
    print("Enjoy your new finds!")



if __name__ == "__main__":
    main()