from audio2text import *
from car import *
from excursion import *
from flights import *
from hotel import *
from image2text import *
from policy import *
from utility import *
from database import *
from agent import part_4_graph

# Ask the user to input a passenger ID, an image URL, or an audio URL
user_input = input("Please enter the passenger ID, the path to the image, or the URL to the audio file: ")

# Determine if the input is an image path, audio URL, or direct ID
if user_input.lower().endswith(('.png', '.jpg', '.jpeg')) or user_input.startswith('http'):
    if user_input.lower().endswith(('.mp3', '.wav', '.flac', '.aac')):
        passenger_id = extract_text_from_audio(user_input)

    else:  
        passenger_id = extract_passenger_id_from_image(user_input)
else:
    # User provided a direct passenger ID
    passenger_id = user_input.strip()

print(f"Using Passenger ID: {passenger_id}")


tutorial_questions = [
    "Hi there, what time is my flight?",
    "Am I allowed to update my flight to something sooner? I want to leave later today.",
    "Update my flight to sometime next week then",
    "The next available option is great",
    "What about lodging and transportation?",
    "Yeah, I think I'd like an affordable hotel for my week-long stay (7 days). And I'll want to rent a car.",
    "OK, could you place a reservation for your recommended hotel? It sounds nice.",
    "Yes, go ahead and book anything that's moderate expense and has availability.",
    "Now for a car, what are my options?",
    "Awesome, let's just get the cheapest option. Go ahead and book for 7 days",
    "Cool, so now what recommendations do you have on excursions?",
    "Are they available while I'm there?",
    "Interesting - I like the museums, what options are there?",
    "OK great, pick one and book it for my second day there."
]

# Assuming 'backup_file' and 'db' are defined elsewhere in your application
shutil.copy(backup_file, db)
thread_id = str(uuid.uuid4())

config = {
    "configurable": {
        "passenger_id": passenger_id,  # Use OCR or STT-generated text
        "thread_id": thread_id,
    },
    "recursion_limit": 100
}

# Debug: Print the configuration to ensure the passenger_id is set correctly
print(f"Configuration: {config}")
from utility import _print_event
_printed = set()
# We can reuse the tutorial questions from part 1 to see how it does.
for question in tutorial_questions:
    events = part_4_graph.stream(
        {"messages": ("user", question)}, config, stream_mode="values"
    )
    for event in events:
        _print_event(event, _printed)
    snapshot = part_4_graph.get_state(config)
    while snapshot.next:
        # We have an interrupt! The agent is trying to use a tool, and the user can approve or deny it
        # Note: This code is all outside of your graph. Typically, you would stream the output to a UI.
        # Then, you would have the frontend trigger a new run via an API call when the user has provided input.
        user_input = input(
           "Do you approve of the above actions? Type 'y' to continue;"
            " otherwise, explain your requested changes.\n\n"
        )
        if user_input.strip().lower() == "y":
            # Just continue
            result = part_4_graph.invoke(None, config)
        else:
            # Satisfy the tool invocation by
            # providing instructions on the requested changes / change of mind
            result = part_4_graph.invoke(
                {
                    "messages": [
                        ToolMessage(
                            tool_call_id=event["messages"][-1].tool_calls[0]["id"],
                            content=f"API call denied by user. Reasoning: '{user_input}'. Continue assisting, accounting for the user's input.",
                        )
                    ]
                },
                config,
            )
        snapshot = part_4_graph.get_state(config)