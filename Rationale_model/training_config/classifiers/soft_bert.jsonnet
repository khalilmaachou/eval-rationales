local is_cose = if std.findSubstr('cose', std.extVar('TRAIN_DATA_PATH')) == [] then false else true;

{
  dataset_reader : {
    type : "rationale_reader" + (if is_cose then '_cose' else ''),
    token_indexers : {
      bert : {
        type : "bert-pretrained",
        pretrained_model : "bert-base-uncased",
        use_starting_offsets: true,
        do_lowercase : true,
        truncate_long_sequences: false
      },
    },
  },
  validation_dataset_reader: {
    type : "rationale_reader" + (if is_cose then '_cose' else ''),
    token_indexers : {
      bert : {
        type : "bert-pretrained",
        pretrained_model : "bert-base-uncased",
        use_starting_offsets: true,
        do_lowercase : true,
        truncate_long_sequences: false
      },
    },
  },
  train_data_path: std.extVar('TRAIN_DATA_PATH'),
  validation_data_path: std.extVar('DEV_DATA_PATH'),
  test_data_path: std.extVar('TEST_DATA_PATH'),
  model: {
    type: "soft_encoder_rationale_model" + (if is_cose then '_cose' else ''),
    text_field_embedder: {
      allow_unmatched_keys: true,
      embedder_to_indexer_map: {
        bert: ["bert", "bert-offsets"],
      },
      token_embedders: {
        bert: {
          type: "bert-pretrained",
          pretrained_model: 'bert-base-uncased',
          requires_grad: '11,10,pooler',
        },
      },
    },
    seq2seq_encoder : {
      type: 'lstm',
      input_size: 768,
      hidden_size: 128,
      num_layers: 1,
      bidirectional: true
    },
    dropout: 0.2,
    attention: {
      type: 'additive',
      vector_dim: 256,
      matrix_dim: 256,
    },
    feedforward_encoder:{
      input_dim: 256,
      num_layers: 1,
      hidden_dims: [128],
      activations: ['relu'],
      dropout: 0.2
    },
    saliency_scorer : {
      type: std.extVar('saliency'),
      threshold: std.extVar('threshold'),
      batch_size: -1
    },
  },
  iterator: {
    type: "bucket",
    sorting_keys: [['document', 'num_tokens']],
    batch_size : std.extVar("batch_size"),
    biggest_batch_first: true
  },
  trainer: {
    num_epochs: 20,
    patience: 5,
    grad_norm: 10.0,
    validation_metric: "+accuracy",
    num_serialized_models_to_keep: 1,
    cuda_device: std.extVar("CUDA_DEVICE"),
    optimizer: {
      type: "adam",
      lr: 2e-5
    }
  },
  random_seed:  std.parseInt(std.extVar("SEED")),
  pytorch_seed: std.parseInt(std.extVar("SEED")),
  numpy_seed: std.parseInt(std.extVar("SEED")),
  evaluate_on_test: true
}
