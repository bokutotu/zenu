use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{parse_macro_input, parse_quote, Data, DeriveInput, Field};

#[proc_macro_derive(ZenuModel, attributes(zenu))]
pub fn my_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let state_dict_impl = impl_state_dict(name, &impl_generics, &ty_generics, where_clause);
    let parameters_impl = impl_parameters(&input);

    let expanded = quote! {
        // #[automatically_derived]
        // #[derive(::serde::Serialize, ::serde::Deserialize)]
        // #[serde(bound(deserialize = "T: ::zenu_matrix::num::Num + ::serde::Deserialize<'de>"))]
        // #modified_struct
        //
        // #state_dict_impl

        #parameters_impl
    };

    TokenStream::from(expanded)
}

fn impl_state_dict(
    name: &syn::Ident,
    impl_generics: &syn::ImplGenerics,
    ty_generics: &syn::TypeGenerics,
    where_clause: Option<&syn::WhereClause>,
) -> TokenStream2 {
    quote! {
        impl #impl_generics ::zenu_layer::StateDict<'de> for #name #ty_generics #where_clause {  }
    }
}

fn impl_parameters(input: &DeriveInput) -> TokenStream2 {
    let name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let fields = match &input.data {
        Data::Struct(data) => &data.fields,
        _ => panic!("ZenuModel only supports structs"),
    };

    // get not #[zenu(skip)] fields
    let fields = fields.iter().filter(|field| !has_zenu_skip_attr(field));

    let weights_code = fields.clone().map(|field| {
        let field_name = &field.ident;
        quote! {
            &self.#field_name.weights()
        }
    });

    let biases_code = fields.clone().map(|field| {
        let field_name = &field.ident;
        quote! {
            &self.#field_name.biases()
        }
    });

    quote!(
        impl #impl_generics ::zenu_layer::Parameteres<T, D> for #name #ty_generics #where_clause {
            fn weights(&self) -> Vec<&::zenu_autograd::Variable<T, D>> {
                let mut params = Vec::new();
                #(
                    let weights = #weights_code;
                    params.extend(weights);
                )*
                params
            }

            fn biases(&self) -> Vec<&::zenu_autograd::Variable<T, D>> {
                let mut params = Vec::new();
                #(
                    let biases = #biases_code;
                    params.extend(biases);
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
