use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{
    parse::{Parse, ParseStream},
    parse_macro_input, parse_quote,
    token::Comma,
    Attribute, Data, DeriveInput, Field, Ident, Token, Type,
};

#[proc_macro_derive(Parameters, attributes(zenu, parameters))]
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

    let (num_type, device_type) = parse_parameters_attr(&input.attrs);

    quote!(
        // impl #impl_generics ::zenu::layer::Parameters #ty_generics for #name #ty_generics #where_clause {
        impl #impl_generics ::zenu::layer::Parameters<#num_type, #device_type> for #name #ty_generics #where_clause {
            fn weights(&self) -> std::collections::HashMap<String, ::zenu::autograd::Variable<#num_type, #device_type>> {
                let mut params = std::collections::HashMap::new();
                #(
                    #weights_code
                )*
                params
            }

            fn biases(&self) -> std::collections::HashMap<String, ::zenu::autograd::Variable<#num_type, #device_type>> {
                let mut params = std::collections::HashMap::new();
                #(
                    #biases_code
                )*
                params
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

fn parse_parameters_attr(attrs: &[Attribute]) -> (Type, Type) {
    let mut num_type: Type = parse_quote!(f32);
    let mut device_type: Type = parse_quote!(Cpu);

    for attr in attrs {
        if attr.path.is_ident("parameters") {
            let args = syn::parse2::<ParametersArgs>(attr.tokens.clone())
                .expect("Failed to parse parameters attribute");
            if let Some(ty) = args.num {
                num_type = ty;
            }
            if let Some(ty) = args.device {
                device_type = ty;
            }
        }
    }

    (num_type, device_type)
}

struct ParametersArgs {
    num: Option<Type>,
    device: Option<Type>,
}

impl Parse for ParametersArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let content;
        syn::parenthesized!(content in input);

        let mut num = None;
        let mut device = None;

        while !content.is_empty() {
            let ident: Ident = content.parse()?;
            let _: Token![=] = content.parse()?;
            let ty: Type = content.parse()?;

            if ident == "num" {
                num = Some(ty);
            } else if ident == "device" {
                device = Some(ty);
            } else {
                return Err(syn::Error::new(
                    ident.span(),
                    "Expected 'num' or 'device' in parameters attribute",
                ));
            }

            if content.peek(Comma) {
                let _: Comma = content.parse()?;
            } else {
                break;
            }
        }

        Ok(ParametersArgs { num, device })
    }
}
