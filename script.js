document.addEventListener('DOMContentLoaded', () => {
    // --- グローバル変数と定数 ---
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    const clearButton = document.getElementById('clear-button');
    const resultDiv = document.getElementById('result');

    let model;
    let isDrawing = false;
    let lastX = 0;
    let lastY = 0;

    // --- 初期化処理 ---
    async function initialize() {
        // スタイル設定
        ctx.strokeStyle = '#000';
        ctx.lineWidth = 20; // AIが認識しやすいように太い線にする
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        
        // 学習済みモデルの読み込み
        try {
            const modelUrl = 'https://storage.googleapis.com/tfjs-models/tfjs/mnist_cnn_v1/model.json';
            model = await tf.loadLayersModel(modelUrl);
            resultDiv.textContent = 'モデル読込完了！数字を描いてください。';
            console.log("Model loaded successfully.");
        } catch (error) {
            console.error("Error loading model:", error);
            resultDiv.textContent = 'モデルの読み込みに失敗しました。';
        }

        // イベントリスナーの設定
        canvas.addEventListener('mousedown', startDrawing);
        canvas.addEventListener('mousemove', draw);
        canvas.addEventListener('mouseup', stopDrawing);
        canvas.addEventListener('mouseout', stopDrawing);
        
        // タッチイベントの対応
        canvas.addEventListener('touchstart', startDrawing);
        canvas.addEventListener('touchmove', draw);
        canvas.addEventListener('touchend', stopDrawing);

        clearButton.addEventListener('click', clearCanvas);
    }

    // --- 描画関連の関数 ---
    function getMousePos(canvasDom, event) {
        const rect = canvasDom.getBoundingClientRect();
        // タッチイベントとマウスイベントで座標の取得方法を分ける
        const clientX = event.clientX || event.touches[0].clientX;
        const clientY = event.clientY || event.touches[0].clientY;
        return {
            x: clientX - rect.left,
            y: clientY - rect.top
        };
    }

    function startDrawing(e) {
        e.preventDefault();
        isDrawing = true;
        const pos = getMousePos(canvas, e);
        [lastX, lastY] = [pos.x, pos.y];
    }

    function draw(e) {
        e.preventDefault();
        if (!isDrawing) return;
        const pos = getMousePos(canvas, e);
        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(pos.x, pos.y);
        ctx.stroke();
        [lastX, lastY] = [pos.x, pos.y];
    }

    function stopDrawing() {
        if (isDrawing) {
            isDrawing = false;
            predict(); // 描画が終わったら予測を実行
        }
    }

    function clearCanvas() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        resultDiv.textContent = 'クリアしました。もう一度どうぞ。';
    }

    // --- AI予測の関数 ---
    async function predict() {
        if (!model) {
            resultDiv.textContent = 'モデルが読み込まれていません。';
            return;
        }

        resultDiv.textContent = '予測中...';

        // tf.tidyを使用してメモリリークを防ぐ
        const prediction = tf.tidy(() => {
            // 1. キャンバスの画像データを取得し、テンソルに変換
            // fromPixelsは画像データをテンソルに変換する。第2引数1はチャンネル数(グレースケール)
            let tensor = tf.browser.fromPixels(canvas, 1);

            // 2. AIモデルの入力サイズ(28x28)にリサイズ
            tensor = tf.image.resizeNearestNeighbor(tensor, [28, 28]);
            
            // 3. データを0-1の範囲に正規化し、バッチ次元を追加
            //  - cast('float32'): データ型を浮動小数点数に変換
            //  - div(tf.scalar(255.0)): 各ピクセルの値を255で割り、0-1の範囲にする
            //  - expandDims(0): モデル入力のために次元を1つ追加 (例: [28, 28, 1] -> [1, 28, 28, 1])
            tensor = tensor.cast('float32').div(tf.scalar(255.0)).expandDims(0);
            
            // 4. 予測を実行
            return model.predict(tensor);
        });

        // 5. 予測結果から最も確率の高いものを取得
        const predictions = await prediction.data();
        prediction.dispose(); // メモリを解放

        const highestPrediction = predictions.indexOf(Math.max(...predictions));
        
        // 6. 結果を表示
        resultDiv.textContent = `これは「${highestPrediction}」です！`;
    }

    // --- 実行開始 ---
    initialize();
});