from flask import Flask, request, jsonify
from deepface import DeepFace
import tempfile
import os
import uuid

app = Flask(__name__)

@app.route('/verify', methods=['POST'])
def verify():
    if 'id_image' not in request.files or 'selfie_image' not in request.files:
        return jsonify({'error': 'Missing images'}), 400
    id_image = request.files['id_image']
    selfie_image = request.files['selfie_image']

    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as id_tmp:
        id_image.save(id_tmp.name)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as selfie_tmp:
        selfie_image.save(selfie_tmp.name)

    try:
        result = DeepFace.verify(
            img1_path=id_tmp.name,
            img2_path=selfie_tmp.name,
            model_name='GhostFaceNet'
        )
        verified = bool(result.get('verified', False))
    except Exception as e:
        os.remove(id_tmp.name)
        os.remove(selfie_tmp.name)
        return jsonify({'error': str(e)}), 500

    os.remove(id_tmp.name)
    os.remove(selfie_tmp.name)

    if verified:
        return jsonify({'verified': True, 'key': str(uuid.uuid4())})
    else:
        return jsonify({'verified': False})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
