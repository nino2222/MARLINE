package Others;

import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.InstanceImpl;

public class MultiSourceInstance extends InstanceImpl {
    protected int data_id;

    public MultiSourceInstance(int data_id, Instance instance){
        super((InstanceImpl) instance);
        this.data_id = data_id;
    }

    public int getDataId() {
            return data_id;
    }

    public void setDataId(int i){
        this.data_id = i;
    }

}
