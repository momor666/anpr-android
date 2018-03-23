package cloud.techstar.carnumberdetector;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.os.AsyncTask;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.util.Base64;
import android.util.Log;
import android.view.MotionEvent;
import android.view.SurfaceView;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import org.json.JSONException;
import org.json.JSONObject;
import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.net.HttpURLConnection;
import java.net.URL;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

public class MainActivity extends AppCompatActivity
        implements CameraBridgeViewBase.CvCameraViewListener2, View.OnTouchListener {

    private static final String TAG = "MainActivity";

    private static final Scalar FACE_RECT_COLOR = new Scalar(0, 255, 0, 255);
    private static final String IMAGE_NAME = "detector-image.png";

    static {
        if (OpenCVLoader.initDebug()) {
            Log.d(TAG, "static initializer: OpenCV loaded");
        } else {
            Log.d(TAG, "static initializer: OpenCV didn't load");
        }
    }

    private JavaCameraView cameraView;
    private TextView textView;
    private ImageView imageView;
    private File mCascadeFile;
    private CascadeClassifier cascadeClassifier;
    private Mat mRgba;
    private Mat mGray;
    private MatWrapper wr;

    private BaseLoaderCallback  mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    try {
                        InputStream is = getResources().openRawResource(R.raw.cascade);
                        File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
                        mCascadeFile = new File(cascadeDir, "cascade.xml");
                        FileOutputStream os = new FileOutputStream(mCascadeFile);

                        byte[] buffer = new byte[4096];
                        int bytesRead;
                        while ((bytesRead = is.read(buffer)) != -1) {
                            os.write(buffer, 0, bytesRead);
                        }
                        is.close();
                        os.close();

                        cascadeClassifier = new CascadeClassifier(mCascadeFile.getAbsolutePath());
                        if (cascadeClassifier.empty()) {
                            Log.e(TAG, "Failed to load cascade classifier");
                            cascadeClassifier = null;
                        } else {
                            Log.i(TAG, "Loaded cascade classifier from " + mCascadeFile.getAbsolutePath());
                        }
                        cascadeDir.delete();
                    } catch (IOException e) {
                        e.printStackTrace();
                        Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
                    }
                    cameraView.enableFpsMeter();
                    cameraView.enableView();
                    cameraView.setOnTouchListener(MainActivity.this);
                    break;
                }
                default: {
                    super.onManagerConnected(status);
                    break;
                }
            }
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        cameraView = (JavaCameraView) findViewById(R.id.java_camera_view);
        cameraView.setVisibility(SurfaceView.VISIBLE);
        cameraView.setCvCameraViewListener(this);
        cameraView.setMaxFrameSize(1280, 720);

        imageView = (ImageView) findViewById(R.id.image_view);
        textView = (TextView) findViewById(R.id.text_view);
        textView.setText(null);
        textView.setTextColor(Color.WHITE);
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (cameraView != null) {
            cameraView.disableView();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (cameraView != null) {
            cameraView.disableView();
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (OpenCVLoader.initDebug()) {
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        } else {
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_2_0, this, mLoaderCallback);
        }
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        mGray = new Mat();
        mRgba = new Mat();
    }

    @Override
    public void onCameraViewStopped() {
        mGray.release();
        mRgba.release();
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();

        MatOfRect currentNumbers = new MatOfRect();
        if (cascadeClassifier != null) {
            cascadeClassifier.detectMultiScale(mGray, currentNumbers, 1.1, 10, 5,
                    new Size(70, 21), new Size(500, 150));
        }

        wr = new MatWrapper();
        for (Rect rect : currentNumbers.toArray()) {
            wr.setImage(mRgba);
            Imgproc.rectangle(mRgba, rect.tl(), rect.br(), FACE_RECT_COLOR, 3);
            Rect rectCropped = new Rect(rect.x, rect.y, rect.width, rect.height);
            wr.getRects().add(rectCropped);
        }
        return mRgba;
    }

    @Override
    public boolean onTouch(View v, MotionEvent event) {
        if (event.getAction() == MotionEvent.ACTION_DOWN) {
            Context context = getApplicationContext();
            int duration = Toast.LENGTH_SHORT;
            String text;
            if (wr.getRects().size() > 0) {
                Mat cropped = new Mat(wr.getImage(), wr.getRects().get(0));
                Bitmap bmp = Bitmap.createBitmap(cropped.cols(), cropped.rows(), Bitmap.Config.ARGB_8888);
                Utils.matToBitmap(cropped, bmp);
                storeToInternalStorage(bmp, IMAGE_NAME);
                String imagePath = context.getFilesDir() + "/" + IMAGE_NAME;
                imageView.setImageBitmap(BitmapFactory.decodeFile(imagePath));

                new CallServer().execute(imagePath);
            } else {
                imageView.setImageResource(0);
                textView.setText(null);
                text = "No detected numbers of photo...";
                Toast toast = Toast.makeText(context, text, duration);
                toast.show();
            }
        }
        return true;
    }

    private void storeToInternalStorage(Bitmap bm, String filename) {
        FileOutputStream outputStream;
        try {
            outputStream = openFileOutput(filename, Context.MODE_PRIVATE);
            bm.compress(Bitmap.CompressFormat.PNG, 85, outputStream);
            outputStream.flush();
            outputStream.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private class MatWrapper {
        private Mat image;
        private List<Rect> rects;

        public Mat getImage() {
            return image;
        }

        public void setImage(Mat image) {
            this.image = image;
        }

        public List<Rect> getRects() {
            return rects;
        }

        public MatWrapper() {
            this.rects = new ArrayList<>();
        }
    }

    private class CallServer extends AsyncTask<String, String, String> {

        private static final String SERVER_URL = "https://diploma-server.herokuapp.com/img/";

        @Override
        protected void onPreExecute() {
            super.onPreExecute();
            textView.setText("Recognition number...");
        }

        @Override
        protected void onPostExecute(String result) {
            super.onPostExecute(result);
            textView.setText(result);
        }

        @Override
        protected String doInBackground(String... params) {
            String encodedImage = convertToBase64(params[0]);
            String jsonBody = createBody("demo" + getCurrentTimeStamp(), encodedImage);
            String res = sendImageToServer(jsonBody);
            return res;
        }

        private String getCurrentTimeStamp() {
            return new SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS").format(new Date());
        }

        private String sendImageToServer(String body) {
            String result = new String();
            try {
                URL url = new URL(SERVER_URL);
                HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                conn.setConnectTimeout(5000);
                conn.setDoOutput(true);
                conn.setDoInput(true);
                conn.setRequestProperty("Content-Type", "application/json");
                conn.setRequestProperty("Accept", "application/json");
                conn.setRequestMethod("POST");

                OutputStreamWriter wr= new OutputStreamWriter(conn.getOutputStream());
                wr.write(body);
                wr.flush();
                wr.close();

                int responseCode = conn.getResponseCode();
                if (responseCode == 200) {
                    StringBuilder resultBuilder = new StringBuilder();
                    InputStream in = new BufferedInputStream(conn.getInputStream());
                    BufferedReader reader = new BufferedReader(new InputStreamReader(in));
                    String line;
                    while ((line = reader.readLine()) != null) {
                        resultBuilder.append(line);
                    }
                    JSONObject jsonObject = new JSONObject(resultBuilder.toString());
                    result = jsonObject.getString("recognition_result");
                }
            } catch (IOException | JSONException e) {
                e.printStackTrace();
            }
            return result;
        }

        private String createBody(String name, String encodedImage) {
            JSONObject body = new JSONObject();
            try {
                body.put("name", name);
                body.put("image", encodedImage);
            } catch (JSONException e) {
                e.printStackTrace();
            }
            return body.toString();
        }

        private String convertToBase64(String path) {
            Bitmap bm = BitmapFactory.decodeFile(path);
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            bm.compress(Bitmap.CompressFormat.PNG, 100, out);
            byte[] byteArrayImage = out.toByteArray();
            String encodedImage = Base64.encodeToString(byteArrayImage, Base64.DEFAULT);
            return encodedImage;
        }
    }

}
