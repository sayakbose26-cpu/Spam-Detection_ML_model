import joblib

mail = input("Enter the mail: ")

model = joblib.load("spam_dection.pk1")

pred_y = model.predict([mail])

print("\n-----------------------")

if pred_y[0] == 1:
    print("SPAM !!")
else:
    print("NOT SPAM ALL OKK")

print("------------------------")