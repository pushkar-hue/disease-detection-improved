# Add these imports at the top of your file
import razorpay
from flask import redirect, url_for, random, render_template, request, jsonify


RAZORPAY_KEY_ID = ""
RAZORPAY_KEY_SECRET = ""

# Initialize Razorpay client
razorpay_client = razorpay.Client(auth=(os.getenv('RAZORPAY_KEY_ID'), os.getenv('RAZORPAY_KEY_SECRET')))

# Add these routes for payment handling

@app.route('/payment', methods=['GET'])
def payment_page():
    """Render the payment options page"""
    return render_template('payment.html')

@app.route('/api/create-order', methods=['POST'])
def create_order():
    """Create a Razorpay order"""
    try:
        data = request.json
        amount = data.get('amount', 500)  # Amount in paise (INR), default is ₹5
        currency = data.get('currency', 'INR')
        receipt = f"receipt_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Create Razorpay Order
        order_data = {
            'amount': amount * 100,  # Convert to paise
            'currency': currency,
            'receipt': receipt,
            'payment_capture': 1  # Auto-capture
        }
        
        order = razorpay_client.order.create(data=order_data)
        
        return jsonify({
            'id': order['id'],
            'amount': order['amount'],
            'currency': order['currency'],
            'key': os.getenv('RAZORPAY_KEY_ID')
        })
    
    except Exception as e:
        print(f"Error creating Razorpay order: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/payment-callback', methods=['POST'])
def payment_callback():
    """Handle Razorpay payment callback"""
    try:
        data = request.form
        payment_id = data.get('razorpay_payment_id', '')
        order_id = data.get('razorpay_order_id', '')
        signature = data.get('razorpay_signature', '')
        
        # Verify the payment signature
        params_dict = {
            'razorpay_payment_id': payment_id,
            'razorpay_order_id': order_id,
            'razorpay_signature': signature
        }
        
        # Verify signature
        razorpay_client.utility.verify_payment_signature(params_dict)
        
        # Update your database with payment info (add your code here)
        # For example:
        # save_payment_details(payment_id, order_id, amount)
        
        # Redirect to success page
        return redirect(url_for('payment_success'))
    
    except Exception as e:
        print(f"Payment verification failed: {str(e)}")
        return redirect(url_for('payment_failure'))

@app.route('/payment-success')
def payment_success():
    """Render payment success page"""
    return render_template('payment_success.html')

@app.route('/payment-failure')
def payment_failure():
    """Render payment failure page"""
    return render_template('payment_failure.html')

# Add a premium report generation function
def generate_premium_report(report_data, image_path, save_path):
    """Generate a premium report with additional insights"""
    # Get enhanced content from Gemini with more detailed analysis
    premium_prompt = f"""
    You are a medical specialist providing an ADVANCED and COMPREHENSIVE analysis.
    
    Based on the {report_data['display_name']} model analysis which classified the image as 
    '{report_data['prediction']}' with {report_data['confidence']:.2f}% confidence, provide:
    
    1. A detailed expert-level analysis with clinical implications
    2. Comprehensive differential diagnoses (at least 3 alternatives)
    3. Detailed treatment protocols and medication options
    4. Latest research findings related to this condition
    5. Prognosis information and long-term management
    6. Specialized referral recommendations
    
    Format this as a PREMIUM medical report with professional medical terminology.
    Include citations to recent medical literature where appropriate.
    """
    
    # Similar to your existing report generation but with premium content
    
    # The rest of your report generation code...
    
    return "Premium report generated"