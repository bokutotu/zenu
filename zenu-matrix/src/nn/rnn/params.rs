pub trait Params {
    type Params;
    fn set_weight(&self, params: &Self::Params);
    fn load_from_params(&mut self, params: &Self::Params);
}
