"""
First author: HongSeok Choi (hongking9@gist.ac.kr)
Corresponding author: Hyunju Lee (hyunjulee@gist.ac.kr)
Code for Early Stopping Based on Unlabeled Samples in Text Classification.
"""
import tensorflow as tf
from tensorflow.keras.layers import Input, Dropout, Dense

class CustomModel(tf.keras.Model):
    def __init__(self,word_freeze=False, *args, **kwargs): 
        self.word_freeze = word_freeze
        super().__init__(*args, **kwargs)
    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        
        ''' We customized tf.keras.Model. So, word embeddings can be freezed or not.The below 
        statement 'self.trainable_variables[1:]' excludes word embeddings from the trainable 
        parameters. In existing pre-trained models, the index for word embeddings is generally 
        set to zero. But, if not, you need to modify the below statement.
        '''
        if self.word_freeze: 
            trainable_vars = self.trainable_variables[1:]
        else:
            trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_pred)
        
        return {m.name: m.result() for m in self.metrics}

class ConstructPtModeler(object):
    def __init__(self, TFModel, Config, pt_model_checkpoint, max_seq_length, num_labels, 
                 dense_dropout_prob=0.2, attention_probs_dropout_prob=0.2, hidden_dropout_prob=0.2, word_freeze=False):
        '''Load a pre-trained model e.g. BERT, RoBERTa, ...'''
        self.num_labels = num_labels
        self.dense_dropout_prob = dense_dropout_prob
        self.word_freeze = word_freeze
        
        self.config = Config.from_pretrained(pt_model_checkpoint, num_labels=num_labels)
        self.config.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.config.hidden_dropout_prob = hidden_dropout_prob

        self.encoder = TFModel.from_pretrained(pt_model_checkpoint, config=self.config, from_pt=True, name="encoder")
        self.init_pt_weights = self.encoder.get_weights()
        
        input_ids = Input(shape=(max_seq_length,), dtype=tf.int32, name="input_ids")
        attention_mask = Input(shape=(max_seq_length,), dtype=tf.int32, name="attention_mask")
        token_type_ids = Input(shape=(max_seq_length,), dtype=tf.int32, name="token_type_ids")
        
        self.enc_inputs = [input_ids, token_type_ids, attention_mask]
        self.enc_output = self.encoder(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
    def final_dense_layer(self):
        '''Define final dense layers'''
        final_output = Dropout(self.dense_dropout_prob)(self.enc_output[0][:,0])
        final_output = Dense(self.num_labels, 
                        kernel_initializer=tf.keras.initializers.TruncatedNormal(
                            stddev=self.config.initializer_range))(final_output)
        return final_output
    def build_model(self):
        '''Initialize and build a model: 
           a pre-trained model + final dense layer '''
        self.encoder.set_weights(self.init_pt_weights)
        final_output = self.final_dense_layer()
        return CustomModel(inputs=self.enc_inputs, outputs=final_output, word_freeze=self.word_freeze)
        #return tf.keras.Model(inputs=self.enc_inputs, outputs=final_output)
