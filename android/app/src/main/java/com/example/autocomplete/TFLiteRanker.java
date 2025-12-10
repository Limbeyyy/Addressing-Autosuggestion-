package com.example.autocomplete;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import org.tensorflow.lite.Interpreter;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.List;
import java.util.Map;
import java.util.ArrayList;

public class TFLiteRanker {
    private Interpreter interpreter;
    private List<String> labels;
    private int maxLen;
    private Map<String,Integer> charMap;

    public TFLiteRanker(AssetManager assets, String modelPath, List<String> labels, Map<String,Integer> charMap, int maxLen) throws IOException {
        this.interpreter = new Interpreter(loadModelFile(assets, modelPath));
        this.labels = labels;
        this.charMap = charMap;
        this.maxLen = maxLen;
    }

    private static MappedByteBuffer loadModelFile(AssetManager assets, String modelPath) throws IOException {
        AssetFileDescriptor fileDescriptor = assets.openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private int[] encode(String s){
        int[] arr = new int[maxLen];
        int i=0;
        for(char c: s.toCharArray()){
            if(i>=maxLen) break;
            Integer idx = charMap.get(String.valueOf(c));
            arr[i++] = idx == null ? 0 : idx;
        }
        while(i<maxLen) arr[i++] = 0;
        return arr;
    }

    // placeholder rank function - implement sorting by score in app code
    public List<String> rankCandidates(String prefix, List<String> candidates, int topK){
        return candidates.subList(0, Math.min(topK, candidates.size()));
    }
}
