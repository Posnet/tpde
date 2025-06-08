// THIS IS MANDATORY - IMPLEMENTS THE TEMPLATE FUNCTION ATTRIBUTE

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, parse::Parser, ItemFn, LitStr, Meta, Lit};

#[proc_macro_attribute]
pub fn template_function(attrs: TokenStream, input: TokenStream) -> TokenStream {
    let input_fn = parse_macro_input!(input as ItemFn);
    
    // MANDATORY: Extract function metadata
    let fn_name = &input_fn.sig.ident;
    let fn_name_str = fn_name.to_string();
    
    // MANDATORY: Parse attributes to get target architecture
    let target = extract_target_from_attrs(attrs);
    
    // MANDATORY: Validate target is one of x86_64-v1, v2, v3, or v4
    assert!(matches!(target.as_str(), "x86_64-v1" | "x86_64-v2" | "x86_64-v3" | "x86_64-v4"),
            "Invalid target: {}. Must be one of x86_64-v1, x86_64-v2, x86_64-v3, x86_64-v4", target);
    
    // MANDATORY: Generate both the function AND metadata
    let metadata_ident = syn::Ident::new(
        &format!("_TPDE_METADATA_{}", fn_name_str.to_uppercase()),
        fn_name.span()
    );
    
    let expanded = quote! {
        // Original function for template extraction
        #[cfg(feature = "template-extraction")]
        #[inline(never)]
        #[no_mangle]
        #input_fn
        
        // Metadata for build script
        #[cfg(feature = "template-extraction")]
        #[link_section = ".tpde_template_metadata"]
        #[used]
        static #metadata_ident: &str = concat!(
            "TPDE_TEMPLATE:",
            #fn_name_str,
            ":",
            #target,
            ";"
        );
    };
    
    TokenStream::from(expanded)
}

// MANDATORY: Helper to extract target architecture
fn extract_target_from_attrs(attrs: TokenStream) -> String {
    let parser = syn::punctuated::Punctuated::<Meta, syn::Token![,]>::parse_terminated;
    let parsed = parser.parse(attrs).unwrap();
    
    for meta in parsed {
        match meta {
            Meta::NameValue(nv) if nv.path.is_ident("target") => {
                if let syn::Expr::Lit(expr_lit) = &nv.value {
                    if let Lit::Str(lit_str) = &expr_lit.lit {
                        return lit_str.value();
                    }
                }
            }
            _ => {}
        }
    }
    
    // Default to "x86_64-v1" if not specified
    "x86_64-v1".to_string()
}

// MANDATORY: Macro for generating encoder dispatch
#[proc_macro]
pub fn dispatch_encoder(input: TokenStream) -> TokenStream {
    let opcode = parse_macro_input!(input as LitStr);
    
    // Generate match arm for instruction dispatch
    let expanded = quote! {
        match #opcode {
            "FAdd" => encode_add_f32(encoder, ctx, inputs, output),
            "FSub" => encode_sub_f32(encoder, ctx, inputs, output),
            "FMul" => encode_mul_f32(encoder, ctx, inputs, output),
            "FDiv" => encode_div_f32(encoder, ctx, inputs, output),
            // ... other opcodes generated from patterns
            _ => Err(EncodingError::UnsupportedInstruction),
        }
    };
    
    TokenStream::from(expanded)
}