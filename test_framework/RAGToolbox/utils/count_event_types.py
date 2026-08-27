import json, os

rootPath = './GUIData/clean'

# def show_event_types():
#     event_types = set()
#     event_type_examples = {}
#     for dir in os.listdir(rootPath):       
#         with open(os.path.join(rootPath, dir, "event_record.jsonl"), "r") as f:
#             eventRecords = json.load(f)
#         for eventRecord in eventRecords:
#             event_type = eventRecord["event_type"]
#             event_types.add(event_type)
#             if event_type not in event_type_examples:
#                 event_type_examples[event_type] = eventRecord  # 记录第一个示例
#     return event_types, event_type_examples

# def main():
#     event_types, event_type_examples = show_event_types()
#     print("Different event types:", event_types)
#     for event_type, example in event_type_examples.items():
#         print(f"Event Type: {event_type}")
#         print(f"Example: {example}")
#         print("-" * 40)

# if __name__ == "__main__":
#     main()
event_type_counts = {}
def count_event_types():
    for dir in os.listdir(rootPath):
        with open(os.path.join(rootPath, dir, "event_record.jsonl"), "r") as f:
            eventRecords = json.load(f)
            for eventRecord in eventRecords:
                event_type = eventRecord["event_type"]
                if event_type in event_type_counts:
                    event_type_counts[event_type] += 1
                else:
                    event_type_counts[event_type] = 1

def main():
    count_event_types()
    print("Event Type Counts:")
    for event_type, count in event_type_counts.items():
        print(f"{event_type}: {count}")

if __name__ == "__main__":
    main()