import joblib
import os

model = joblib.load(os.path.join('backend', 'model.pkl'))
tfidf = joblib.load(os.path.join('backend', 'tfidf.pkl'))
le = joblib.load(os.path.join('backend', 'label_encoder.pkl'))
print('model type:', type(model))
print('label encoder classes:', list(le.classes_))
print('label mapping:', {c: int(le.transform([c])[0]) for c in le.classes_})
print('model classes:', getattr(model, 'classes_', 'MISSING'))
print('n classes:', getattr(model, 'n_classes_', 'MISSING'))
print('estimators:', getattr(model, 'n_estimators', 'MISSING'))

for text in [
    'python sql aws data scientist',
    'ethical hacking cybersecurity analyst',
    'database administration java developer',
    'networking linux cybersecurity',
    'flask javascript web developer'
]:
    vec = tfidf.transform([text])
    pred = model.predict(vec)[0]
    probs = model.predict_proba(vec)[0]
    print(text, '=>', le.inverse_transform([pred])[0], pred, 'probs', probs)
