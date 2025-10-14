def filter_questions_by_dataset_ids(dataset, questions):
    dataset_ids = {str(item["id"]) for item in dataset}
    filtered_questions = [q for q in questions if str(q["id"]) in dataset_ids]
    return filtered_questions

def sort_by_id(data):
    return sorted(data, key=lambda x: int(x["id"]))
