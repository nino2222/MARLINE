package Others;

import moa.streams.ArffFileStream;

public class MSfileStream {

    private int num_of_data;
    private ArffFileStream[] streams;

    public MSfileStream(String data_path){
        this.num_of_data = 1;
        this.streams = new ArffFileStream[this.num_of_data];

        for (int i = 0; i < this.num_of_data; i++){
            this.streams[i] = new ArffFileStream(data_path, -1);
            this.streams[i].prepareForUse();
        }
    }

    public MSfileStream(String[] data_paths){
        this.num_of_data = data_paths.length;
        this.streams = new ArffFileStream[this.num_of_data];

        for (int i = 0; i < this.num_of_data; i++){
            this.streams[i] = new ArffFileStream(data_paths[i], -1);
            this.streams[i].prepareForUse();
        }
    }

    public boolean hasMoreInstances(){
        for (ArffFileStream stream: streams){
            if (stream.hasMoreInstances())
                return true;
        }
        return false;
    }

    public MultiSourceInstance nextInstance(){
        for (int i = 0; i < streams.length; i++){
            while (streams[i].hasMoreInstances()){
                return new MultiSourceInstance(i, streams[i].nextInstance().getData());
            }
        }
        return null;
    }

    public ArffFileStream[] getStreams(){
        return this.streams;
    }
}
