from transformers.pipelines import TextGenerationPipeline
from .rep_control_reading_vec import WrappedReadingVecModel

class RepControlPipeline(TextGenerationPipeline):
    def __init__(self, 
                 model, 
                 tokenizer, 
                 layers, 
                 block_name="decoder_block", 
                 control_method="reading_vec",
                 **kwargs):
        
        # TODO: implement different control method and supported intermediate modules for different models
        assert control_method == "reading_vec", f"{control_method} not supported yet"
        assert block_name == "decoder_block" or "LlamaForCausalLM" in model.config.architectures, f"{model.config.architectures} {block_name} not supported yet"
        self.wrapped_model = WrappedReadingVecModel(model, tokenizer)
        self.wrapped_model.unwrap()
        self.wrapped_model.wrap_block(layers, block_name=block_name)
        self.block_name = block_name
        self.layers = layers
        self.probs = []

        super().__init__(model=model, tokenizer=tokenizer, **kwargs)
   
    def __call__(self, text_inputs, activations=None, **kwargs):
        self.probs = []

        if activations is not None:
            self.wrapped_model.reset()
            self.wrapped_model.set_controller(self.layers, activations, self.block_name)

        outputs = super().__call__(text_inputs, **kwargs)
        self.wrapped_model.reset()
        if len(outputs) == 1:
            outputs[0]['probs'] = self.probs

        return outputs
    
    # overwrite _forward method to get the scores for probs output
    def _forward(self, model_inputs, **kwargs):
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs.get("attention_mask", None)
        # Allow empty prompts
        if input_ids.shape[1] == 0:
            input_ids = None
            attention_mask = None
            in_b = 1
        else:
            in_b = input_ids.shape[0]
        prompt_text = model_inputs.pop("prompt_text")

        # Get num of token parameter if exist and remove it from kwargs, default to 5
        num_top_tokens = 5
        if "num_top_tokens" in kwargs and kwargs["num_top_tokens"]:
            num_top_tokens = kwargs["num_top_tokens"]
            kwargs.pop("num_top_tokens")
        
        # If there is a prefix, we may need to adjust the generation length. Do so without permanently modifying
        # generate_kwargs, as some of the parameterization may come from the initialization of the pipeline.
        prefix_length = kwargs.pop("prefix_length", 0)
        if prefix_length > 0:
            has_max_new_tokens = "max_new_tokens" in kwargs or (
                "generation_config" in kwargs
                and kwargs["generation_config"].max_new_tokens is not None
            )
            if not has_max_new_tokens:
                kwargs["max_length"] = kwargs.get("max_length") or self.model.config.max_length
                kwargs["max_length"] += prefix_length
            has_min_new_tokens = "min_new_tokens" in kwargs or (
                "generation_config" in kwargs
                and kwargs["generation_config"].min_new_tokens is not None
            )
            if not has_min_new_tokens and "min_length" in kwargs:
                kwargs["min_length"] += prefix_length

        # BS x SL
        generated_sequence = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

        scores = None
        if "output_scores" in kwargs and kwargs["output_scores"]:
            import torch
            scores = generated_sequence.scores
            generated_sequence = generated_sequence.sequences
            selected_tokens = generated_sequence.squeeze(0).tolist()
            selected_tokens = selected_tokens[len(input_ids[0]):]
            selected_token_strings = self.tokenizer.convert_ids_to_tokens(selected_tokens)            
            for score, selected_token in zip(scores, selected_token_strings):
                probabilities = torch.nn.functional.softmax(score[0], dim=0)
                top_tokens = torch.topk(probabilities, num_top_tokens)
                token_strings = self.tokenizer.convert_ids_to_tokens(top_tokens.indices)
                probs = top_tokens.values.tolist()
                return_dict = {}
                return_dict["selected_token"] = selected_token.replace("_", "")
                for tokenstr, prob in zip(token_strings,probs):
                    return_dict[tokenstr.replace("_", "")] = prob
                self.probs.append(return_dict)

        out_b = generated_sequence.shape[0]
        if self.framework == "pt":
            generated_sequence = generated_sequence.reshape(in_b, out_b // in_b, *generated_sequence.shape[1:])
        elif self.framework == "tf":
            generated_sequence = tf.reshape(generated_sequence, (in_b, out_b // in_b, *generated_sequence.shape[1:]))
        return {"generated_sequence": generated_sequence, "input_ids": input_ids, "prompt_text": prompt_text}