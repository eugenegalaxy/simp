import os

from simp.submodules.deepface.deepface import DeepFace


base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# img1 = os.path.join(base_path, 'images/new_entries/arnold/Arnold_Schwarzenegger_0003.jpg')
# img3 = os.path.join(base_path, 'images/mom.jpg')
# img2 = os.path.join(base_path, 'images/manual_database/Hugo-sv/Hugo.jpeg')
# img4 = os.path.join(base_path, 'images/mysql_database/Jevgenijs_Galaktionovs/Jevgenijs_Galaktionovs_9.jpg')
# db_path = os.path.join(base_path, 'images/manual_database')


img1 = '/home/eugenegalaxy/Desktop/for_identity_verification/eugene_clean.jpg'
img2 = '/home/eugenegalaxy/Desktop/for_identity_verification/eugene_abomination.png'
img3 = '/home/eugenegalaxy/Documents/projects/simp/simp/tests/deepface_dataset/bo/mask_off.jpg'
img4 = '/home/eugenegalaxy/Desktop/for_identity_verification/hugo_abomination.png'
img5 = '/home/eugenegalaxy/Desktop/for_identity_verification/andreas_abomination.png'
img6 = '/home/eugenegalaxy/Desktop/for_identity_verification/andreas_clean.jpg'
# result  = DeepFace.verify(img1, img3)
# results = DeepFace.verify([[img5, img6], [img5, img1], [img5, img3]])
# print(results)


# df = DeepFace.find(img_path=img1, db_path=db_path, enforce_detection=False)
# print(df)

obj = DeepFace.analyze(img_path=img3, actions=['age', 'gender'])
# objs = DeepFace.analyze(["img1.jpg", "img2.jpg", "img3.jpg"]) #analyzing multiple faces same time
print(obj["age"], " years old ", " ", obj["gender"])
