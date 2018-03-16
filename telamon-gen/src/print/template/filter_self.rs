let values = self::filter({{>choice.arg_names choice}}ir_instance, store);
self::restrict({{>choice.arg_ids choice}}ir_instance, store, values, diff)?;
