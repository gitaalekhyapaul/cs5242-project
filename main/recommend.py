# Stage 0: Our Data Catalog
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
    main_category = get_choice(list(CATALOG.keys()))

    while True:
        # Stage 2: Show Genres based on Stage 1
        print(f"\n--- {main_category} Genres ---")
        genres = list(CATALOG[main_category].keys())
        selected_genre = get_choice(genres)

        # Stage 3: Show Items in Genre
        print(f"\n--- Best {selected_genre} {main_category} ---")
        items = CATALOG[main_category][selected_genre]
        item_choice = get_choice(items)

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