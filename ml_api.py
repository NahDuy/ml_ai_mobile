from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os
from pymongo import MongoClient
from datetime import datetime, timedelta
from dotenv import load_dotenv

app = Flask(__name__)

# Sử dụng trực tiếp URI MongoDB
MONGO_URI = "mongodb+srv://loan:21112003loanhoang@cluster0.1sdjxcz.mongodb.net/shoper?retryWrites=true&w=majority&appName=Cluster0"

# Kết nối MongoDB
client = MongoClient(MONGO_URI)
db = client["shoper"]
product_collection = db["products"]
review_collection = db["reviewproduct"]

# Load mô hình ML từ đường dẫn cụ thể (mới nhất)
model = joblib.load("C:\\Users\\FPT SHOP\\ecommerce_microservices\\ML\\final_trending_model.pkl")


@app.route("/predict-trending-from-db", methods=["GET"])
def predict_trending_from_db():
    try:
        # Lấy tất cả sản phẩm từ MongoDB
        raw_products = list(product_collection.find({}))

        print(">>> Số sản phẩm lấy được:", len(raw_products))
        if not raw_products:
            return jsonify({"message": "Không có sản phẩm"}), 404

        df = pd.DataFrame(raw_products)
        df["product_id"] = df["_id"].astype(str)

        # Điền mặc định nếu thiếu cột
        df["name"] = df.get("name", "")
        df["rating_avg"] = df.get("rating_avg", 0)
        df["sale_quantity"] = df.get("sale_quantity", 0)
        df["view_count"] = df.get("view_count", 0)

        # Dự đoán bằng mô hình ML mới
        X = df[["rating_avg", "sale_quantity", "view_count"]]
        proba = model.predict_proba(X)[:, 1]
        df["score"] = proba

        print(">>> Phân bố điểm dự đoán:", df["score"].describe())

        # Trả về top 4 sản phẩm có điểm cao nhất
        top_products = df.sort_values(by="score", ascending=False).head(4)[["product_id", "name", "score"]]
        return top_products.to_json(orient="records", force_ascii=False)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
