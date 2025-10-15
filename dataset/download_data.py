import json
import re
import wikipediaapi

wiki = wikipediaapi.Wikipedia(language='en', user_agent='MyApp/1.0 (contact@example.com)')

def clean_text(text):
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]+', ' ', text)
    text = text.replace('"', "'")
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def get_category_articles(category_name):
    category_page = wiki.page("Category:" + category_name)
    articles = []
    for title, member in category_page.categorymembers.items():
        if member.ns == 0:
            articles.append(member)
    return articles

categories = [
    "Computer science", "Mathematics", "Physics", "Biology", "Chemistry", "History", 
    "Geography", "Economics", "Politics", "Philosophy", "Literature", "Art", "Music", 
    "Film", "Television", "Sports", "Astronomy", "Medicine", "Psychology", "Sociology", 
    "Law", "Religion", "Architecture", "Engineering", "Linguistics", "Anthropology", 
    "Education", "Agriculture", "Business", "Finance", "Environmental science", "Ecology", 
    "Zoology", "Botany", "Geology", "Meteorology", "Oceanography", "Statistics", "Ethnography", 
    "Archaeology", "Astrophysics", "Biochemistry", "Genetics", "Neuroscience", "Microbiology", 
    "Virology", "Informatics", "Robotics", "Nanotechnology", "Quantum mechanics", "Thermodynamics", 
    "Classical mechanics", "Relativity", "Optics", "Acoustics", "Fluid dynamics", "Materials science", 
    "Nuclear physics", "Particle physics", "Organic chemistry", "Inorganic chemistry", 
    "Analytical chemistry", "Physical chemistry", "Space exploration", "Cosmology", "Cartography", 
    "Demography", "Probability", "Logic", "Semiotics", "Ethics", "Metaphysics", "Epistemology", 
    "Aesthetics", "Folklore", "Mythology", "Poetry", "Drama", "Fiction", "Non-fiction", 
    "Biography", "Memoir", "Journalism", "Photography", "Sculpture", "Painting", "Drawing", 
    "Ceramics", "Interior design", "Landscape architecture", "Urban planning", "Transportation", 
    "Military history", "Ancient history", "Medieval history", "Modern history", "Contemporary history", 
    "World history", "Regional history", "Cultural studies", "Media studies", "Communication", 
    "Social media", "Public relations", "Advertising", "Marketing", "Management", "Leadership", 
    "Entrepreneurship", "Accounting", "Taxation", "Investment", "Insurance", "Real estate", 
    "Supply chain management", "Logistics", "Operations research", "Industrial engineering", 
    "Mechanical engineering", "Electrical engineering", "Civil engineering", "Chemical engineering", 
    "Aerospace engineering", "Software engineering", "Computer programming", "Web development", 
    "Database", "Cybersecurity", "Information technology", "Data science", "Artificial intelligence",
    "Machine learning", "Deep learning", "Natural language processing", "Computer graphics", 
    "Virtual reality", "Augmented reality", "Game development", "Video games", "Board games", 
    "Football", "Basketball", "Baseball", "Soccer", "Tennis", "Golf", "Cricket", "Rugby", 
    "Athletics", "Swimming", "Gymnastics", "Cycling", "Hockey", "Skiing", "Snowboarding", 
    "Skating", "Boxing", "Wrestling", "Martial arts", "Fencing", "Sailing", "Rowing", 
    "Equestrian", "Diving", "Surfing", "Triathlon", "Paralympic sports", "Esports", 
    "Notable people", "Awards", "Contests", "Festivals", "Holidays", "Customs", "Traditions", 
    "Cuisine", "Food", "Cooking", "Recipes", "Beverages", "Wine", "Beer", "Spirits", 
    "Farming", "Horticulture", "Forestry", "Fishing", "Hunting", "Wildlife", "Conservation", 
    "Climate change", "Renewable energy", "Sustainability", "Environmentalism", "Natural disasters", 
    "Geopolitics", "International relations", "Law enforcement", "Criminology", "Forensic science", 
    "Bioinformatics", "Computational biology", "Cognitive science", "Human-computer interaction",
    "Speech recognition", "Computer vision", "Blockchain", "Quantum computing", "Edge computing",
    "Internet of Things", "Space technology", "Comparative literature", "Cultural anthropology",
    "Gender studies", "Postcolonial studies", "Queer theory", "Critical theory", "Translation studies",
    "Theology", "Human rights", "Development studies", "Urban sociology", "Behavioral economics",
    "Game theory", "Organizational behavior", "Labor economics", "Social psychology", 
    "Conflict resolution", "Graphic design", "Fashion design", "Animation", "Comic books",
    "Street art", "Calligraphy", "Tattoo art", "Graffiti", "Epidemiology", "Public health",
    "Medical ethics", "Geriatrics", "Palliative care", "Radiology", "Oncology", "Dentistry",
    "Veterinary medicine", "Minimalism", "Zero waste", "Sustainable fashion", "Tiny house movement",
    "Digital nomadism", "Home automation", "Distance learning", "Educational psychology",
    "Instructional design", "Curriculum development", "Special education", "Career development"
]

articles_dict = {}

for category in categories:
    print(f"Processing category: {category}")
    articles = get_category_articles(category)
    for article in articles:
        if article.title not in articles_dict:
            articles_dict[article.title] = (article, category)

dataset = []
for idx, (title, (article, category)) in enumerate(articles_dict.items()):
    print(f"Processing article: {title}")
    try:
        if not article.exists():
            print(f"Skipping missing article: {title}")
            continue
        cleaned_text = clean_text(article.text)
        if not cleaned_text:
            print(f"Skipping empty article: {title}")
            continue
        dataset.append({"id": idx, "category": category, "title": title, "text": cleaned_text})
    except Exception as e:
        print(f"Error processing article '{title}': {e}")

with open('dataset/dataset_with_categories.json', 'w', encoding='utf-8') as f:
    json.dump(dataset, f, ensure_ascii=False, indent=4)

print("Dataset created successfully with", len(dataset), "entries!")
