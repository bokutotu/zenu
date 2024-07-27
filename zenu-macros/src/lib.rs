use proc_macro::TokenStream;
use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::quote;
use syn::{parse_macro_input, Attribute, Data, DeriveInput, Field, Ident, Lit, Meta, NestedMeta};

#[proc_macro_derive(ZenuModel, attributes(zenu))]
pub fn zenu_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    let (nums, devices) = input
        .clone()
        .attrs
        .into_iter()
        .filter_map(extract_bounds)
        .next()
        .unwrap_or_default();

    // let state_dict = impl_state_dict(&input, &nums, &devices);
    let parameters_impl = impl_parameters(&input);

    let expanded = quote! {
        // #state_dict
        #parameters_impl
    };

    TokenStream::from(expanded)
}

// /// Extracts numeric and device bounds from a zenu attribute
// fn extract_bounds(attr: Attribute) -> Option<(Vec<String>, Vec<String>)> {
//     match attr.parse_meta() {
//         Ok(Meta::List(meta_list)) if meta_list.path.is_ident("zenu") => {
//             let mut num_bounds = Vec::new();
//             let mut device_bounds = Vec::new();
//
//             for nested in meta_list.nested {
//                 if let NestedMeta::Meta(Meta::List(inner_list)) = nested {
//                     if inner_list.path.is_ident("bound") {
//                         for inner_nested in inner_list.nested {
//                             if let NestedMeta::Meta(Meta::NameValue(name_value)) = inner_nested {
//                                 if let Lit::Str(lit_str) = &name_value.lit {
//                                     let values: Vec<String> = lit_str
//                                         .value()
//                                         .split(',')
//                                         .map(str::trim)
//                                         .map(String::from)
//                                         .collect();
//
//                                     if name_value.path.is_ident("num") {
//                                         num_bounds = values;
//                                     } else if name_value.path.is_ident("device") {
//                                         device_bounds = values;
//                                     }
//                                 }
//                             }
//                         }
//                     }
//                 }
//             }
//
//             Some((num_bounds, device_bounds))
//         }
//         _ => None,
//     }
// }
//
// fn impl_state_dict(input: &DeriveInput, nums: &[String], devices: &[String]) -> TokenStream2 {
//     let name = &input.ident;
//     let (_, ty_generics, where_clause) = input.generics.split_for_impl();
//
//     let nums_trait_bound = nums.iter().map(|num| {
//         let num = Ident::new(num, Span::call_site());
//         quote! {
//             #num: ::zenu_matrix::num::Num + ::serde::Serialize + ::serde::Deserialize<'de>
//         }
//     });
//
//     let devices_trait_bound = devices.iter().map(|device| {
//         let device = Ident::new(device, Span::call_site());
//         quote! {
//             #device: ::zenu_matrix::device::Device + ::serde::Serialize + ::serde::Deserialize<'de>
//         }
//     });
//
//     let bounds: Vec<_> = nums_trait_bound.chain(devices_trait_bound).collect();
//
//     let bounds_quote = if !bounds.is_empty() {
//         quote! { #(#bounds),* }
//     } else {
//         quote! {}
//     };
//
//     quote! {
//         impl <'de, #bounds_quote> ::zenu_layer::StateDict<'de> for #name #ty_generics #where_clause {}
//     }
// }

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
        impl #impl_generics ::zenu_layer::Parameters<T, D> for #name #ty_generics #where_clause {
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
