from models.tisasrec import TiSASRec
import torch
from pathlib import Path
import json
import time
from pprint import pprint

import numpy as np

device = torch.device("cuda" \
    if torch.cuda.is_available() else \
    # "mps" \
    # if torch.backends.mps.is_available() else \
    "cpu"
)

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
MAX_ITEMS_DISPLAYED = 12
MAX_LEN = 50
TIME_SPAN = 256

def generate_time_matrix(time_seq, time_span):
    size = time_seq.shape[0]
    time_matrix = np.zeros([size, size], dtype=np.int32)
    for i in range(size):
        for j in range(size):
            time_matrix[i][j] = min(abs(time_seq[i]-time_seq[j]), time_span)
    return time_matrix

def personalize_time_sequence(timestamps):
    values = [int(timestamp) for timestamp in timestamps]
    if not values:
        return []

    time_diffs: list[int] = []
    for current_time, next_time in zip(values, values[1:]):
        diff = next_time - current_time
        if diff < 0:
            raise ValueError("Timestamps must be sorted by user before time normalization.")
        if diff > 0:
            time_diffs.append(diff)

    time_scale = min(time_diffs) if time_diffs else 1
    time_min = min(values)
    return [
        int(round((timestamp - time_min) / time_scale) + 1)
        for timestamp in values
    ]

def pad_categories(categories, max_len):
    padded = np.zeros(max_len, dtype=np.int64)
    trimmed = categories[-max_len:]
    if trimmed:
        padded[-len(trimmed) :] = trimmed
    return padded

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
                return options[choice - 1], choice - 1
            print(f"Please pick a number between 1 and {len(options)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def main():
    selected_items = []
    print("--- Welcome to the App & Game Discovery Tool ---")

    # Stage 1: Initial Category Selection
    print("\nWhat are you interested in today?")
    main_category, _ = get_choice(MAIN_CATEGORIES)

    #* default to mobilerec settings
    settings = {
        "num_items": 10173,
        "num_categories": 48,
    }
    model_checkpoint = Path("metadata") / "mobilerec_model.pt"

    #* switch to steamrec settings
    if main_category == MAIN_CATEGORIES[1]:
        settings = {
            "num_items": 9598,
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
        max_len=MAX_LEN,
        time_span=TIME_SPAN,
        hidden_size=128,
        num_blocks=2,
        num_heads=2,
        dropout=0.2,
        device=device,
    ).to(device)

    model.load_state_dict(torch.load(model_checkpoint, map_location=device))
    input_ids = []
    metadata_seq = []
    category_seq = []
    time_seq = []

    while True:
        # Stage 2: Show subcategories based on Stage 1
        print(f"\n--- {main_category} Categories ---")
        subcategories = list(categories.keys())[:MAX_ITEMS_DISPLAYED]
        selected_subcategory, _ = get_choice(subcategories)

        # Stage 3: Show Items in subcategory
        print(f"\n--- Category: {selected_subcategory} ({main_category}) ---")
        item_ids = categories[selected_subcategory]["item_ids"][:MAX_ITEMS_DISPLAYED]
        item_names = [metadata.get(str(item_id))["app_name"] for item_id in item_ids]
        item_choice, choice_idx = get_choice(item_names)

        selection_time = int(time.time())

        selected_item_id = item_ids[choice_idx]
        selected_item = metadata.get(str(selected_item_id))

        input_ids.append(selected_item_id)
        metadata_seq.append([
            1, # positive review
            0, # no review upvotes
            selected_item.get("num_reviews"),
            selected_item.get("avg_rating"),
            selected_item.get("price"),
        ])
        category_seq.append(
            pad_categories(
                selected_item.get("category_ids", []),
                settings["num_categories"],
            ),
        )
        time_seq.append(selection_time)


        if item_choice not in selected_items:
            selected_items.append(item_choice)
            print(f"Added '{item_choice}' to your list!")
        else:
            print(f"'{item_choice}' is already in your list.")

        # Stage 4 & 5: Loop or Finish
        print("\nWould you like to add more?")
        continue_choice, _ = get_choice([f"Yes, add more {main_category.lower()}", "No, I'm finished"])

        if continue_choice == "No, I'm finished":
            break
        clear_screen()

    print(np.array(metadata_seq).shape)
    print(np.array(category_seq).shape)

    scores = model.score_all_items(
        input_ids=torch.from_numpy(np.array(input_ids)).unsqueeze(0).to(device),
        metadata_seq=torch.from_numpy(np.array(metadata_seq)).float().unsqueeze(0).to(device),
        category_seq=torch.from_numpy(np.array(category_seq)).to(device),
        time_matrix=torch.from_numpy(
            generate_time_matrix(
                np.array(personalize_time_sequence(time_seq)),
                TIME_SPAN
            ),
        ).unsqueeze(0).to(device),
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