use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{parse_macro_input, Data, DeriveInput, Field};

#[proc_macro_derive(Parameters, attributes(zenu))]
pub fn zenu_derive_parameters(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    let parameters_impl = impl_parameters(&input);

    TokenStream::from(parameters_impl)
}

fn impl_parameters(input: &DeriveInput) -> TokenStream2 {
    let name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let fields = match &input.data {
        Data::Struct(data) => &data.fields,
        _ => panic!("ZenuModel only supports structs"),
    };

    let fields = fields.iter().filter(|field| !has_zenu_skip_attr(field));

    let weights_code = fields.clone().map(|field| {
        let field_name = &field.ident;
        quote! {
            for (name, variable) in &self.#field_name.weights() {
                let name = format!("{}.{}", stringify!(#field_name), name);
                params.insert(name.clone(), variable.clone());
            }
        }
    });

    let biases_code = fields.clone().map(|field| {
        let field_name = &field.ident;
        quote! {
            for (name, variable) in &self.#field_name.biases() {
                let name = format!("{}.{}", stringify!(#field_name), name);
                params.insert(name.clone(), variable.clone());
            }
        }
    });

    let load_parameters_code = fields.map(|field| {
        let field_name = &field.ident;
        quote! {
            let filed_name_str= stringify!(#field_name);
            let field_parameters: std::collections::HashMap<String, ::zenu_autograd::Variable<T, D>> = parameters
                .clone()
                .into_iter()
                .filter(|(name, _)| name.starts_with(&format!("{}.", filed_name_str)))
                .map(|(name, variable)| {
                    let name = name.split(".").collect::<Vec<&str>>();
                    let name = name[1..].join(".");
                    (name, variable)
                })
                .collect();
            self.#field_name.load_parameters(field_parameters.clone());
        }
    });

    quote!(
        impl #impl_generics ::zenu_layer::Parameters #ty_generics for #name #ty_generics #where_clause {
            fn weights(&self) -> std::collections::HashMap<String, ::zenu_autograd::Variable<T, D>> {
                let mut params = std::collections::HashMap::new();
                #(
                    #weights_code;
                )*
                params
            }

            fn biases(&self) -> std::collections::HashMap<String, ::zenu_autograd::Variable<T, D>> {
                let mut params = std::collections::HashMap::new();
                #(
                    #biases_code;
                )*
                params
            }
            fn load_parameters(&self, parameters: std::collections::HashMap<String, ::zenu_autograd::Variable<T, D>>) {
                #(
                    #load_parameters_code
                )*
            }
        }
    )
}

fn has_zenu_skip_attr(field: &Field) -> bool {
    field
        .attrs
        .iter()
        .any(|attr| attr.path.is_ident("zenu") && attr.tokens.to_string().contains("skip"))
}
