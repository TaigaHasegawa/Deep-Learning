import torch
from torch import nn
import torchvision
import pretrainedmodels
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    
    def __init__(self,embed_dim,image_size=14):
        super(Encoder,self).__init__()

        #pretrained ImageNet ResNet-152
        resnet = torchvision.models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_dim)
        self.pool = nn.AdaptiveAvgPool2d(1)
        """
        #The following for pnasnet
        # model_name = 'pnasnet5large'
        # model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        #model=model.features
        # self.model = model
        """

    def forward(self,x):
        with torch.no_grad():
            h = self.resnet(x)
        h=self.pool(h)
        h = h.reshape(h.size(0), -1)
        h = self.linear(h)
        #if you use pnasnet use the following
        #h=self.model(x)
        h=h.unsqueeze(1) #(batch_size,1,embed_dim)
        return h



class Decoder(nn.Module):
    def __init__(self,embed_dim,decoder_dim,vocab_size):
        """
        #:param embed_dim: embedding size
        #:param decoder_dim: size of decoder's RNN
        #:param vocab_size: size of vocabulary
        #:param encoder_dim: feature size of encoded images
        """ 
        super(Decoder,self).__init__()

        self.embed_dim=embed_dim
        self.decoder_dim=decoder_dim
        self.vocab_size=vocab_size

        
        self.embedding=nn.Embedding(vocab_size,embed_dim)

        self.decode_step=nn.LSTMCell(embed_dim,decoder_dim,bias=True) #used as StatefulLSTM

        self.dropout = nn.Dropout(p=0.5)

        self.init_h=nn.Linear(embed_dim,decoder_dim)
        self.init_c=nn.Linear(embed_dim,decoder_dim)


        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        #Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)
    
    def load_pretrained_embeddings(self, embeddings):
        """
        #Loads embedding layer with pre-trained embeddings.
        #:param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def init_hidden_state(self, encoder_out):
        """
        #Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        #:param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        #:return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).
        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def forward(self,encoder_out,encoded_captions,caption_lengths):
        """
        #Forward propagation.
        #:param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        #:param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        #:param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        #:return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths
        caption_lengths, sort_idx = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_idx]
        encoded_captions = encoded_captions[sort_idx]

        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        
        encoder_out=encoder_out.view(encoder_out.size(0),-1)
        h,c=self.decode_step(encoder_out,(h,c))

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            h, c = self.decode_step(
                embeddings[:batch_size_t, t, :],
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds

        return predictions, encoded_captions, decode_lengths,  sort_idx