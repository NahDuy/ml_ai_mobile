import pandas as pd
import joblib
from pymongo import MongoClient
from datetime import datetime
import traceback

# Load model
model = joblib.load("C:\\Users\\FPT SHOP\\ecommerce_microservices\\ML\\final_trending_model.pkl")

# MongoDB setup
MONGO_URI = "mongodb+srv://loan:21112003loanhoang@cluster0.1sdjxcz.mongodb.net/shoper?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(MONGO_URI)
db = client["shoper"]
product_collection = db["products"]

def update_trending_scores():
    try:
        print(f"[{datetime.now()}] >>> Bắt đầu cập nhật trending score...")

        # Chỉ lấy trường cần thiết
        products = list(product_collection.find({}, {
            "_id": 1,
            "rating_avg": 1,
            "sale_quantity": 1,
            "view_count": 1
        }))

        if not products:
            print(">>> Không có sản phẩm nào.")
            return

        df = pd.DataFrame(products)
        df["rating_avg"] = df.get("rating_avg", 0).fillna(0)
        df["sale_quantity"] = df.get("sale_quantity", 0).fillna(0)
        df["view_count"] = df.get("view_count", 0).fillna(0)

        X = df[["rating_avg", "sale_quantity", "view_count"]]
        df["trending_score"] = model.predict_proba(X)[:, 1]

        for idx, row in df.iterrows():
            product_collection.update_one(
                {"_id": products[idx]["_id"]},
                {"$set": {"trending_score": float(row["trending_score"])}}
            )

        print(f"[{datetime.now()}] ✅ Đã cập nhật trending_score cho {len(df)} sản phẩm.")

    except Exception as e:
        print("❌ Lỗi:", e)
        traceback.print_exc()

if __name__ == "__main__":
    update_trending_scores()
