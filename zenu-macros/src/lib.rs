use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{parse_macro_input, parse_quote, spanned::Spanned, Data, DeriveInput, Field};

#[proc_macro_derive(ZenuModel, attributes(zenu))]
pub fn my_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let modified_struct = modify_struct(&input);
    let state_dict_impl = impl_state_dict(name, &impl_generics, &ty_generics, where_clause);
    let parameters_impl = impl_parameters(&input);

    let expanded = quote! {
        #impl_serde

        #state_dict_impl

        #parameters_impl
    };

    TokenStream::from(expanded)
}

fn impl_serde(input: &DeriveInput) -> TokenStream2 {
    let name = &input.ident;
    let generics = &input.generics;
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let fields = match &input.data {
        Data::Struct(data) => &data.fields,
        _ => panic!("ZenuModel only supports structs"),
    };

    let modified_fields = fields.iter().map(|f| {
        let mut field = f.clone();
        if has_zenu_skip_attr(&field) {
            // Replace #[zenu(skip)] with #[serde(skip)]
            field.attrs.retain(|attr| !attr.path.is_ident("zenu"));
            let serde_skip = parse_quote!(#[serde(skip)]);
            field.attrs.push(serde_skip);
        }
        field
    });

    quote! {
        #[derive(Serialize, Deserialize)]
        #[serde(bound(deserialize = "T: Num + Deserialize<'de>"))]
        struct #name #generics #where_clause {
            #(#modified_fields,)*
        }
    }
}

fn impl_state_dict(
    name: &syn::Ident,
    impl_generics: &syn::ImplGenerics,
    ty_generics: &syn::TypeGenerics,
    where_clause: Option<&syn::WhereClause>,
) -> TokenStream2 {
    quote! {
        impl #impl_generics StateDict<'de> for #name #ty_generics #where_clause {
            // StateDict is already implemented, so we don't need to add anything here
        }
    }
}

fn impl_parameters(input: &DeriveInput) -> TokenStream2 {
    let name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let fields = match &input.data {
        Data::Struct(data) => &data.fields,
        _ => panic!("ZenuModel only supports structs"),
    };

    let weight_fields = fields.iter().filter_map(|f| {
        if is_weight_field(f) && !has_zenu_skip_attr(f) {
            let name = &f.ident;
            Some(quote! { &self.#name, })
        } else {
            None
        }
    });

    let bias_fields = fields.iter().filter_map(|f| {
        if is_bias_field(f) && !has_zenu_skip_attr(f) {
            let name = &f.ident;
            Some(quote! { &self.#name, })
        } else {
            None
        }
    });

    quote! {
        impl #impl_generics Parameters<T, D> for #name #ty_generics #where_clause {
            fn weights(&self) -> Vec<&Variable<T, D>> {
                vec![#(#weight_fields)*]
            }

            fn biases(&self) -> Vec<&Variable<T, D>> {
                vec![#(#bias_fields)*]
            }
        }
    }
}

fn has_zenu_skip_attr(field: &Field) -> bool {
    field
        .attrs
        .iter()
        .any(|attr| attr.path.is_ident("zenu") && attr.tokens.to_string().contains("skip"))
}

fn is_weight_field(field: &Field) -> bool {
    field
        .ident
        .as_ref()
        .map_or(false, |ident| ident.to_string().contains("weight"))
}

fn is_bias_field(field: &Field) -> bool {
    field
        .ident
        .as_ref()
        .map_or(false, |ident| ident.to_string().contains("bias"))
}
