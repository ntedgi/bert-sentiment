from data_utils import build_tokenizer, build_embedding_matrix, Tokenizer4Bert, ABSADataset

PATH = "/home/naort/Desktop/Sentiment/ABSA-PyTorch/datasets/semeval14/israel_Test.xml.seg"
tokenizer = Tokenizer4Bert(80, "bert-base-uncased")

left = "Tsunami warning signs being placed on beaches throughout country  "
right = "#HolyLand via jpost.com"
aspect = "israel"
ABSAD = ABSADataset(PATH, tokenizer, False)
data = ABSAD.prapare_data(aspect, 1, left, right, tokenizer)
print(data)
