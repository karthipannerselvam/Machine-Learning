from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)

# Sample dataset for customer transactions
data = {
    'customer_id': [1, 2, 1, 3, 4, 5, 6, 7, 8],
    'product_id': ['water', 'idli', 'juice', 'soda', 'water', 'soda', 'juice', 'idli', 'water'],
    'quantity': [2, 43, 5, 10, 8, 12, 3, 6, 7],
    'amount': [1000, 2, 300, 500, 1500, 400, 350, 100, 1200],
    'discount': [10, 34, 5, 20, 15, 25, 10, 8, 12],
    'total_amount': [990, -32, 285, 480, 1485, 375, 340, 92, 1188]
}
df = pd.DataFrame(data)

# Initial hypothesis template
initial_hypothesis = ['?', '?', '?', '?']


# Helper function to update hypothesis
def update_hypothesis(hypothesis, example):
    new_hypothesis = hypothesis[:]
    for i in range(len(hypothesis)):
        if hypothesis[i] == '?':
            new_hypothesis[i] = example[i]
        elif hypothesis[i] != example[i]:
            new_hypothesis[i] = '?'
    return new_hypothesis

# Find-S algorithm to generate the hypothesis
def find_s_algorithm(data):
    hypothesis = initial_hypothesis[:]
    for index, row in data.iterrows():
        if row['total_amount'] > 0:
            example = [row['quantity'], row['amount'], row['discount'], row['total_amount']]
            hypothesis = update_hypothesis(hypothesis, example)
    return hypothesis

# Recommendation function based on hypothesis
def get_recommendations(new_customer_id, hypothesis):
    customer_data = df[df['customer_id'] == new_customer_id]
    if customer_data.empty:
        return "No data available for this customer ID."

    bought_products = set(customer_data['product_id'])
    all_products = set(df['product_id'])
    products_to_recommend = list(all_products - bought_products)

    recommendations = []
    for product in products_to_recommend:
        product_data = df[df['product_id'] == product].iloc[0]
        example = [product_data['quantity'], product_data['amount'], product_data['discount'], product_data['total_amount']]

        match = True
        for i in range(len(hypothesis)):
            if hypothesis[i] != '?' and hypothesis[i] != example[i]:
                match = False
                break
        if match:
            recommendations.append(product)

    if not recommendations:
        return "No products to recommend."

    return recommendations

# Precompute the hypothesis
hypothesis = find_s_algorithm(df)

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    error_message = ""
    if request.method == 'POST':
        try:
            customer_id = int(request.form['customer_id'])
            recommendations = get_recommendations(customer_id, hypothesis)
            if isinstance(recommendations, str):
                error_message = recommendations
                recommendations = []
        except ValueError:
            error_message = "Invalid customer ID."

    return render_template('index.html', recommendations=recommendations, error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
